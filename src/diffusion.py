import logging

import hydra
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.optim import AdamW

from utils.callbacks import MetricTracker, create_evaluation_callback
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


class DiffusionModel(pl.LightningModule):
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
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

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
    datamodule = GenerativeDataModule(cfg, log)
    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    data = datamodule.get_original_data()

    # Initialize model
    log.info("Initializing diffusion model...")
    model = DiffusionModel(cfg.model)

    # Find appropriate values
    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = torch.cuda.device_count()
        devices = 1
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
    eval_callback = create_evaluation_callback(
        log, cfg, model_type="diffusion", evaluation_level="standard"
    )
    trainer = pl.Trainer(
        max_epochs=cfg.main.max_epochs,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        callbacks=[tracker, eval_callback],
        enable_progress_bar=True,
        log_every_n_steps=10,
        gradient_clip_val=cfg.main.grad_clip,
    )
    trainer.fit(model, datamodule)
    log.info("Training complete.")

    # Generate samples
    log.info("Generating samples...")
    model.eval()
    device = next(model.parameters()).device

    if (
        cfg.main.dataset.lower() == "two_moons"
        or cfg.main.dataset.lower() == "2d_gaussians"
    ):
        generated_samples = model.sample((2000, 2), device)

        X = data.cpu().numpy()  # Move original data to CPU for plotting
        samples = (
            generated_samples.cpu().numpy()
        )  # Move generated samples to CPU for plotting

        save_2d_samples(samples, X, tracker, "diffusion", cfg.main.dataset.lower())

        visualize_diffusion_process(model, generated_samples)
    else:
        # Use DDIM sampler for large dataset so it doesn't take a million years
        final_samples = model.ddim_sample((16, 3, 256, 256), device)

        # Save generated samples
        save_image_samples(final_samples, "diffusion", cfg.main.dataset.lower())
        plot_loss_function(tracker, "diffusion", cfg.main.dataset.lower())

    fig = plot_evaluation_metrics(
        eval_callback,
        save_path=f"diffusion_{cfg.main.dataset}_metrics_dashboard.png",
    )
    plt.close()
    # Get summary table
    summary_df = create_metrics_summary_table(eval_callback)
    print(summary_df)


if __name__ == "__main__":
    """
    Main execution: train the diffusion model and visualize results

    This implementation demonstrates:
    1. Forward diffusion q(x_t | x_0) with cosine noise schedule
    2. Reverse diffusion p_θ(x_{t-1} | x_t) using a simple U-Net
    3. Training with the simplified denoising objective
    4. Sample generation through iterative denoising
    5. Visualization of original vs generated distributions
    """
    main()
