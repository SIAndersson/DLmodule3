import logging
import os
from typing import Dict

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import MLFlowLogger
from torch.optim import AdamW

from utils.callbacks import (
    EvaluationMixin,
    MetricTracker,
    create_early_stopping_callback,
    create_evaluation_config,
    create_model_checkpoint_callback,
)
from utils.dataset import GenerativeDataModule
from utils.models import CNN, MLP
from utils.seeding import set_seed
from utils.visualisation import (
    create_metrics_summary_table,
    plot_evaluation_metrics,
    plot_loss_function,
    save_2d_samples,
    save_image_samples,
    visualize_diffusion_process,
)

sns.set_theme(style="whitegrid", context="talk", font="DejaVu Sans")

log = logging.getLogger(__name__)

if torch.cuda.is_available():
    # Get properties of the first available GPU
    device_props = torch.cuda.get_device_properties(0)
    if device_props.major >= 7:
        torch.set_float32_matmul_precision("high")
        print("Tensor cores enabled globally")


class DiffusionModel(pl.LightningModule, EvaluationMixin):
    """
    Denoising Diffusion Probabilistic Model (DDPM) implementation

    Mathematical Framework:

    Forward Process (adding noise):
    q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)
    q(x_t | x_0) = N(x_t; √(α̅_t) x_0, (1-α̅_t) I)

    Where:
    - β_t is the noise schedule
    - α_t = 1 - β_t
    - α̅_t = ∏_{s=1}^t α_s (cumulative product)

    Reverse Process (denoising):
    p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), σ_t^2 I)

    Training Objective:
    L = E_{x_0, ε, t} [||ε - ε_θ(√(α̅_t) x_0 + √(1-α̅_t) ε, t)||^2]
    """

    def __init__(
        self,
        model_cfg: DictConfig,
        evaluator_config: Dict,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.input_dim = model_cfg.input_dim
        self.num_timesteps = model_cfg.num_steps
        self.lr = model_cfg.lr
        self.hidden_dim = model_cfg.hidden_dim
        self.num_layers = model_cfg.num_layers
        self.time_embed_dim = model_cfg.time_embed_dim
        self.weight_decay = model_cfg.weight_decay
        # Store metrics history
        self.metrics_history = {
            "epoch": [],
            "fid": [],
            "wasserstein_distance": [],
            "mmd": [],
            "coverage": [],
            "precision": [],
            "js_divergence": [],
            "energy_distance": [],
            "density_ks_stat": [],
            "log_density_ratio": [],
            "mode_collapse_score": [],
            "duplicate_ratio": [],
            "mean_pairwise_distance": [],
            "min_pairwise_distance": [],
            "std_pairwise_distance": [],
            "distance_entropy": [],
        }
        self.val_metrics = []

        # Initialize the denoising network
        if model_cfg.model_type.upper() == "MLP":
            self.model = MLP(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                time_embed_dim=self.time_embed_dim,
                model_type="noise_predictor",
            )
        elif model_cfg.model_type.upper() == "CNN":
            self.model = CNN(
                input_channels=3,  # Assuming RGB images
                time_embed_dim=self.time_embed_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                model_type="noise_predictor",
            )
        else:
            raise ValueError(f"Unknown model type: {model_cfg.model_type}")

        self.setup_evaluation(evaluator_config)

        # Pre-compute diffusion schedule parameters
        self.register_buffer("betas", self._cosine_beta_schedule(self.num_timesteps))
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, dim=0))
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - self.alphas_cumprod)
        )

        # For reverse process
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / self.alphas))

    def setup(self, stage=None):
        self.setup_evaluator(stage)

    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """
        Cosine noise schedule from "Improved Denoising Diffusion Probabilistic Models"

        Mathematical formulation:
        α̅_t = cos²((t/T + s)π/2 / (1 + s))
        β_t = 1 - α_t = 1 - α̅_t/α̅_{t-1}

        This schedule provides better sample quality than linear schedules.
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = (
            torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        )
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion: sample x_t from q(x_t | x_0)

        Mathematical formula:
        x_t = √(α̅_t) * x_0 + √(1 - α̅_t) * ε

        Where ε ~ N(0, I) is Gaussian noise

        This is the reparameterization trick applied to:
        q(x_t | x_0) = N(x_t; √(α̅_t) x_0, (1-α̅_t) I)
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        if x_start.dim() == 2:
            # 2D data case (e.g., two moons, 2D Gaussian)
            reshape_dims = (-1, 1)
        elif x_start.dim() == 4:
            # 4D data case (e.g., images)
            reshape_dims = (-1, 1, 1, 1)
        else:
            # General case: reshape to match all dimensions except batch
            reshape_dims = (-1,) + (1,) * (x_start.dim() - 1)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(*reshape_dims)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(
            *reshape_dims
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, x_start, t, noise=None):
        """
        Compute the denoising loss

        Mathematical objective:
        L_simple = E_t,x_0,ε [||ε - ε_θ(x_t, t)||²]

        This is the simplified loss from the DDPM paper, which works better
        than the full variational lower bound.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        # Forward process: add noise to get x_t
        x_noisy = self.q_sample(x_start, t, noise)

        # Predict the noise using our model
        predicted_noise = self.model(x_noisy, t)

        # MSE loss between actual and predicted noise
        loss = F.mse_loss(noise, predicted_noise)
        return loss

    def on_train_start(self):
        log.info(f"[GPU {self.trainer.local_rank}] Using device: {self.device}")

    def training_step(self, batch, batch_idx):
        """Training step: sample timestep and compute denoising loss"""
        x = (
            batch[0] if isinstance(batch, (list, tuple)) else batch
        )  # Assume batch is (data, labels) or just (data,)
        batch_size = x.shape[0]

        # Randomly sample timestep for each example
        # Uniform sampling: t ~ Uniform(0, T-1)
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=x.device)

        loss = self.p_losses(x, t)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step - compute evaluation metrics."""
        # Only run evaluation on the first batch (very intensive)
        if batch_idx == 0:
            val_metrics = self.run_evaluation()

            if self.trainer.world_size > 1:
                self.trainer.strategy.barrier()

            for k, v in val_metrics.items():
                self.log(f"eval/{k}", v, sync_dist=True, on_epoch=True)

            if self.trainer.world_size > 1:
                self.trainer.strategy.barrier()

            self.val_metrics.append(val_metrics)

    def on_validation_epoch_end(self):
        if self.val_metrics and any(len(d) > 0 for d in self.val_metrics):
            self.metrics_history["epoch"].append(self.current_epoch)
            # Get all keys from the first dict (since all dicts have the same keys)
            keys = self.val_metrics[0].keys()
            for key in keys:
                # Collect all values for this key across all dicts
                values = [d[key] for d in self.val_metrics]
                # Compute the mean
                mean_value = torch.mean(
                    torch.tensor(values, dtype=torch.float32)
                ).item()
                # Append the mean to the history
                self.metrics_history[key].append(mean_value)
        self.val_metrics.clear()

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def get_metrics_history(self):
        return self.metrics_history

    @torch.no_grad()
    def p_sample(self, x, t):
        """
        Single reverse diffusion step: sample x_{t-1} from p_θ(x_{t-1} | x_t)

        Mathematical formulation:
        μ_θ(x_t, t) = 1/√α_t * (x_t - β_t/√(1-α̅_t) * ε_θ(x_t, t))

        Then: x_{t-1} = μ_θ(x_t, t) + σ_t * z, where z ~ N(0, I)
        """
        if x.dim() == 2:
            # 2D data case (e.g., two moons, 2D Gaussian)
            reshape_dims = (-1, 1)
        elif x.dim() == 4:
            # 4D data case (e.g., images)
            reshape_dims = (-1, 1, 1, 1)
        else:
            # General case: reshape to match all dimensions except batch
            reshape_dims = (-1,) + (1,) * (x.dim() - 1)

        betas_t = self.betas[t].reshape(*reshape_dims)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(
            *reshape_dims
        )
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].reshape(*reshape_dims)

        # Predict noise and compute mean
        predicted_noise = self.model(x, t)
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )

        if t.min() == 0:
            # No noise at t=0
            return model_mean
        else:
            # Add noise for t > 0
            noise = torch.randn_like(x)
            # Use fixed variance σ_t² = β_t (as in original DDPM)
            return model_mean + torch.sqrt(betas_t) * noise

    @torch.no_grad()
    def sample(self, shape, device):
        """
        Full reverse diffusion: generate samples by iterating from T to 0

        Algorithm:
        1. Start with x_T ~ N(0, I)
        2. For t = T-1, ..., 0: x_t = p_sample(x_{t+1}, t+1)
        3. Return x_0
        """
        # Start from pure noise
        x = torch.randn(shape, device=device)

        # Reverse diffusion process
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t)

        return x

    @torch.no_grad()
    def ddim_step(self, x_t, t, prev_t, eta=0.0):
        """
        Single DDIM step (inspired by https://nn.labml.ai/diffusion/stable_diffusion/sampler/ddim.html, but simplified and torchified)

        """
        batch_size = x_t.shape[0]
        t_batch = torch.full((batch_size,), t, device=x_t.device, dtype=torch.long)
        prev_t_batch = torch.full(
            (batch_size,), prev_t, device=x_t.device, dtype=torch.long
        )

        # Handle dimensions for broadcasting
        if x_t.dim() == 2:
            reshape_dims = (-1, 1)
        elif x_t.dim() == 4:
            reshape_dims = (-1, 1, 1, 1)
        else:
            reshape_dims = (-1,) + (1,) * (x_t.dim() - 1)

        # Get alpha values
        alpha_t = self.alphas_cumprod[t_batch].reshape(*reshape_dims)
        alpha_prev = (
            self.alphas_cumprod[prev_t_batch].reshape(*reshape_dims)
            if prev_t > 0
            else torch.ones_like(alpha_t)
        )

        # Predict noise
        predicted_noise = self.model(x_t, t_batch)

        # Predict x_0
        pred_x0 = (x_t - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(
            alpha_t
        )

        # Compute direction to x_t
        dir_xt = (
            torch.sqrt(1 - alpha_prev - eta**2 * (1 - alpha_t / alpha_prev))
            * predicted_noise
        )

        # Compute x_{t-1}
        x_prev = torch.sqrt(alpha_prev) * pred_x0 + dir_xt

        # Add stochastic noise if eta > 0
        if eta > 0 and prev_t > 0:
            noise = torch.randn_like(x_t)
            sigma = (
                eta
                * torch.sqrt((1 - alpha_prev) / (1 - alpha_t))
                * torch.sqrt(1 - alpha_t / alpha_prev)
            )
            x_prev += sigma.reshape(*reshape_dims) * noise

        return x_prev

    @torch.no_grad()
    def ddim_sample(
        self, shape, device, num_inference_steps: int = 50, eta: float = 0.0
    ):
        """
        DDIM sampling (https://arxiv.org/pdf/2010.02502)

        Args:
            shape: Shape of samples to generate
            device: Device to generate on
            num_inference_steps: Number of denoising steps (much less than num_timesteps)
            eta: DDIM parameter (0.0 = deterministic, DDIM, 1.0 = stochastic, DDPM-like)
        """
        # Create subset of timesteps
        timesteps = torch.linspace(
            self.num_timesteps - 1, 0, num_inference_steps, dtype=torch.long
        )
        timesteps = timesteps.to(device)

        # Start from pure noise
        x = torch.randn(shape, device=device)

        # Reverse diffusion with skipped timesteps
        for i, t in enumerate(timesteps):
            # Get next timestep (or 0 if we're at the end)
            prev_t = (
                timesteps[i + 1]
                if i + 1 < len(timesteps)
                else torch.tensor(0, device=device)
            )

            x = self.ddim_step(x, t, prev_t, eta)

        return x


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """
    Complete training pipeline with visualization
    """
    # Set random seeds for reproducibility
    set_seed(cfg.main.seed)

    # Create dataset
    if cfg.main.gradient_accumulation:
        gradient_accumulation = cfg.main.batch_size // 32
    else:
        gradient_accumulation = 1
    datamodule = GenerativeDataModule(cfg, log)
    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    data = datamodule.get_original_data()

    # Initialize model
    log.info("Initializing diffusion model...")
    eval_config = create_evaluation_config(
        log, cfg, model_type="diffusion", evaluation_level=cfg.main.evaluation_level
    )
    model = DiffusionModel(cfg.model, eval_config)

    # Find appropriate values
    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = torch.cuda.device_count()
        strategy = "auto"
    elif torch.backends.mps.is_available():
        accelerator = "mps"
        devices = 1
        strategy = "auto"
    else:
        accelerator = "cpu"
        devices = "auto"
        strategy = "auto"

    # Train model
    log.info("Training model...")
    tracker = MetricTracker()
    model_checkpoint_callback = create_model_checkpoint_callback(
        model_name="diffusion", dataset_type=cfg.main.dataset.lower()
    )
    early_stopping_callback = create_early_stopping_callback(patience=50)

    # Initialise MLFlowLogger (wanted to try this for a while so this is me indulging)
    experiment_name = f"sweep_diffusion_{cfg.main.dataset.lower()}"
    project_root = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )  # Go up to project root
    mlruns_path = os.path.join(project_root, "mlruns")

    # Ensure mlruns directory exists
    os.makedirs(mlruns_path, exist_ok=True)
    mlflow_logger = MLFlowLogger(
        experiment_name=experiment_name,
        tracking_uri=f"file:{mlruns_path}",
        log_model=False,
    )
    # Log hyperparameters
    mlflow_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    trainer = pl.Trainer(
        max_epochs=cfg.main.max_epochs,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        callbacks=[tracker, model_checkpoint_callback],
        logger=mlflow_logger,
        enable_progress_bar=True,
        log_every_n_steps=10,
        gradient_clip_val=cfg.main.grad_clip,
        accumulate_grad_batches=gradient_accumulation,
        num_sanity_val_steps=0,
    )
    try:
        trainer.fit(model, datamodule)
        log.info("Training complete.")
    # Most likely fails due to Cuda OOM, return high values for loss and metric
    except Exception as e:
        log.error(f"Training failed: {e}")
        return 1e10, 1e10

    log.info(f"Train losses: {tracker.train_losses}")

    # Generate samples
    log.info("Generating samples...")
    model.eval()
    device = next(model.parameters()).device

    if cfg.main.visualization:
        model_checkpoint_callback.best_model_path
        best_model = DiffusionModel.load_from_checkpoint(
            model_checkpoint_callback.best_model_path
        )
        if (
            cfg.main.dataset.lower() == "two_moons"
            or cfg.main.dataset.lower() == "2d_gaussians"
        ):
            generated_samples = best_model.sample((2000, 2), device)

            X = data.cpu().numpy()  # Move original data to CPU for plotting
            samples = (
                generated_samples.cpu().numpy()
            )  # Move generated samples to CPU for plotting

            save_2d_samples(samples, X, tracker, "diffusion", cfg.main.dataset.lower())

            visualize_diffusion_process(best_model, generated_samples)
        else:
            # Use DDIM sampler for large dataset so it doesn't take a million years
            final_samples = best_model.ddim_sample((16, 3, 256, 256), device)

            # Save generated samples
            save_image_samples(final_samples, "diffusion", cfg.main.dataset.lower())
            plot_loss_function(tracker, "diffusion", cfg.main.dataset.lower())

    metrics_history = model.get_metrics_history()
    if cfg.main.visualization:
        fig = plot_evaluation_metrics(
            metrics_history,
            "diffusion",
            save_path=f"diffusion_{cfg.main.dataset}_metrics_dashboard.png",
        )
        plt.close()

    # Get summary table
    summary_df = create_metrics_summary_table(metrics_history)
    print(summary_df)

    # First check that we don't have inf or NaN
    final_train_loss = tracker.train_losses[-1]
    final_fid = metrics_history["fid"][-1]
    if not np.isfinite(final_train_loss) or not np.isfinite(final_fid):
        last_finite_train_idx = np.where(np.isfinite(tracker.train_losses))[0][-1]
        last_finite_fid_idx = np.where(np.isfinite(metrics_history["fid"]))[0][-1]
        final_train_loss = tracker.train_losses[last_finite_train_idx]
        final_fid = metrics_history["fid"][last_finite_fid_idx]

    mlflow_logger.experiment.log_metric(
        mlflow_logger.run_id, "final_train_loss", final_train_loss
    )
    mlflow_logger.experiment.log_metric(mlflow_logger.run_id, "final_fid", final_fid)

    # Return train loss and eval/coverage
    return final_train_loss, final_fid


if __name__ == "__main__":
    main()
