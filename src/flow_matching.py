import logging
import os
from typing import Dict, Optional

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
)

# Configure matplotlib for better aesthetics
sns.set_theme(style="whitegrid", context="talk", font="DejaVu Sans")

log = logging.getLogger(__name__)

if torch.cuda.is_available():
    # Get properties of the first available GPU
    device_props = torch.cuda.get_device_properties(0)
    if device_props.major >= 7:
        torch.set_float32_matmul_precision("high")
        log.info("Tensor cores enabled globally")


class FlowMatching(pl.LightningModule, EvaluationMixin):
    """
    Flow Matching implementation following Lipman et al. (2023).

    Mathematical Foundation:
    Flow Matching learns a vector field v_θ(x, t) that generates a continuous
    normalizing flow (CNF) from a simple prior p_0 (e.g., Gaussian) to the
    target data distribution p_1.

    The key insight is to match the vector field to the conditional vector field:
    u_t(x) = (x_1 - x_0) where x_t = (1-t)x_0 + tx_1 (linear interpolation)

    This avoids the expensive computation of the log-determinant Jacobian
    required in standard CNFs.
    """

    def __init__(
        self,
        model_cfg: DictConfig,
        evaluator_config: Dict,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.sigma = model_cfg.sigma  # Noise level for conditional flow matching
        self.ode_solver = model_cfg.ode_solver
        self.num_steps = model_cfg.num_steps
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

        # Vector field network v_θ(x, t)
        if model_cfg.model_type.upper() == "MLP":
            self.vector_field = MLP(
                input_dim=model_cfg.input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                time_embed_dim=self.time_embed_dim,
                model_type="vector_field",
            )
            self.dim = model_cfg.input_dim
        elif model_cfg.model_type.upper() == "CNN":
            self.vector_field = CNN(
                input_channels=3,  # Assuming RGB images
                time_embed_dim=self.time_embed_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                model_type="vector_field",
            )
            self.dim = (3, 256, 256)
        else:
            raise ValueError(f"Unknown model type: {model_cfg.model_type}")

        self.setup_evaluation(evaluator_config)

    def setup(self, stage=None):
        self.setup_evaluator(stage)

    def conditional_vector_field(
        self, x_t: torch.Tensor, x_1: torch.Tensor, x_0: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the target conditional vector field u_t(x).

        Mathematical Derivation:
        For the linear interpolant x_t = (1-t)x_0 + tx_1, we have:
        u_t(x) = d/dt x_t = x_1 - x_0

        This is the velocity field that transforms x_0 to x_1 linearly.

        Args:
            x_t: Interpolated samples at time t
            x_1: Target data samples
            x_0: Source noise samples
            t: Time values

        Returns:
            Conditional vector field u_t(x)
        """
        return x_1 - x_0

    def sample_interpolant(
        self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Sample from the interpolating path between x_0 and x_1.

        Mathematical Context:
        We use the linear interpolant: x_t = (1-t)x_0 + tx_1
        This creates a straight-line path in data space from noise to data.

        Alternative interpolants (e.g., trigonometric) can be used but
        linear interpolation is simple and empirically effective.
        """
        t = t.view(-1, *([1] * (x_0.dim() - 1)))  # Broadcast time dimension
        return (1 - t) * x_0 + t * x_1

    def flow_matching_loss(self, x_1: torch.Tensor) -> torch.Tensor:
        """
        Compute the Flow Matching loss.

        Mathematical Formulation:
        L_FM(θ) = E_{t~U[0,1], x_0~p_0, x_1~p_1} [||v_θ(x_t, t) - u_t(x_t)||²]

        where:
        - t ~ U[0,1]: uniform time sampling
        - x_0 ~ p_0: samples from prior (Gaussian noise)
        - x_1 ~ p_1: samples from data distribution
        - x_t = (1-t)x_0 + tx_1: interpolant
        - u_t(x_t) = x_1 - x_0: target vector field

        This loss trains the network to predict the direction from noise to data.
        """
        batch_size = x_1.shape[0]
        device = x_1.device

        # Sample uniform time t ∈ [0, 1]
        t = torch.rand(batch_size, device=device)

        # Sample from prior p_0 (standard Gaussian with added noise)
        # Mathematical note: Adding σ·ε provides stability and prevents
        # the model from overfitting to exact interpolation paths
        x_0 = torch.randn_like(x_1)
        if self.sigma > 0:
            x_0 = x_0 + self.sigma * torch.randn_like(x_1)

        # Compute interpolant x_t = (1-t)x_0 + tx_1
        x_t = self.sample_interpolant(x_0, x_1, t)

        # Predict vector field v_θ(x_t, t)
        v_pred = self.vector_field(x_t, t)

        # Compute target vector field u_t(x_t) = x_1 - x_0
        v_target = self.conditional_vector_field(x_t, x_1, x_0, t)

        # MSE loss between predicted and target vector fields
        loss = F.mse_loss(v_pred, v_target)

        return loss

    def euler_step(self, x: torch.Tensor, t: float, dt: float) -> torch.Tensor:
        """
        Single Euler integration step.

        Mathematical Formulation:
        x_{n+1} = x_n + dt * v_θ(x_n, t_n)

        This is the simplest numerical ODE solver with O(dt) local error.
        Fast but less accurate than higher-order methods.
        """
        t_tensor = torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype)
        v = self.vector_field(x, t_tensor)
        return x + dt * v

    def rk4_step(self, x: torch.Tensor, t: float, dt: float) -> torch.Tensor:
        """
        Single RK4 (Runge-Kutta 4th order) integration step.

        Mathematical Formulation:
        k1 = v_θ(x_n, t_n)
        k2 = v_θ(x_n + dt/2 * k1, t_n + dt/2)
        k3 = v_θ(x_n + dt/2 * k2, t_n + dt/2)
        k4 = v_θ(x_n + dt * k3, t_n + dt)
        x_{n+1} = x_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

        This 4th-order method has O(dt^4) local error, making it much more
        accurate than Euler for smooth vector fields, at the cost of 4x
        more function evaluations per step.
        """

        def get_v(x_curr, t_curr):
            t_tensor = torch.full(
                (x_curr.shape[0],), t_curr, device=x_curr.device, dtype=x_curr.dtype
            )
            return self.vector_field(x_curr, t_tensor)

        k1 = get_v(x, t)
        k2 = get_v(x + dt / 2 * k1, t + dt / 2)
        k3 = get_v(x + dt / 2 * k2, t + dt / 2)
        k4 = get_v(x + dt * k3, t + dt)

        return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def sample(
        self, num_samples: int, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Generate samples by solving the ODE dx/dt = v_θ(x, t) from t=0 to t=1.

        Mathematical Context:
        Starting from x_0 ~ p_0 (prior), we integrate the learned vector field:
        dx/dt = v_θ(x, t) for t ∈ [0, 1]

        This transforms samples from the prior to the data distribution.
        The choice of ODE solver affects accuracy vs. computational cost:
        - Euler: Fast, O(dt) error, good for real-time applications
        - RK4: Slower, O(dt^4) error, better for high-quality samples
        """
        if device is None:
            device = next(self.parameters()).device

        # Start from prior samples x_0 ~ N(0, I)
        if isinstance(self.dim, int):
            x = torch.randn(num_samples, self.dim, device=device)
        else:
            x = torch.randn(num_samples, *self.dim, device=device)

        # Integration parameters
        dt = 1.0 / self.num_steps

        # Choose integration method
        if self.ode_solver == "euler":
            step_fn = self.euler_step
        elif self.ode_solver == "rk4":
            step_fn = self.rk4_step
        else:
            raise ValueError(f"Unknown ODE solver: {self.ode_solver}")

        # Integrate from t=0 to t=1
        with torch.no_grad():
            for i in range(self.num_steps):
                t = i * dt
                x = step_fn(x, t, dt)

        return x

    def on_train_start(self):
        log.info(f"[GPU {self.trainer.local_rank}] Using device: {self.device}")

    def training_step(self, batch, batch_idx):
        """Training step - compute flow matching loss."""
        x_1 = batch[0] if isinstance(batch, (list, tuple)) else batch
        loss = self.flow_matching_loss(x_1)

        self.log("train_loss", loss, prog_bar=True, sync_dist=True, on_epoch=True)
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
        """Configure Adam optimizer."""
        return AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def get_metrics_history(self):
        return self.metrics_history

    def fast_sample(
        self,
        num_samples: int,
        device: Optional[torch.device] = None,
        solver: str = "euler",
        eval_steps: int = 10,
    ) -> torch.Tensor:
        original_steps = self.num_steps
        original_solver = self.ode_solver

        self.num_steps = eval_steps
        self.ode_solver = solver

        samples = self.sample(num_samples, device=device)

        # Restore original settings
        self.num_steps = original_steps
        self.ode_solver = original_solver

        return samples


# Example usage and testing
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Set random seed for reproducibility
    set_seed(cfg.main.seed)

    log.info("Initializing Flow Matching model...")
    # Initialize model
    eval_config = create_evaluation_config(
        log, cfg, model_type="vector_field", evaluation_level=cfg.main.evaluation_level
    )
    model = FlowMatching(cfg.model, eval_config)

    # Create sample data
    log.info("Setting up dataset...")
    if cfg.main.gradient_accumulation:
        gradient_accumulation = cfg.main.batch_size // 32
    else:
        gradient_accumulation = 1
    datamodule = GenerativeDataModule(cfg, log)
    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    X_train = datamodule.get_original_data()

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

    # Initialize PyTorch Lightning Trainer and fit the model
    log.info("Training model...")
    tracker = MetricTracker()
    model_checkpoint_callback = create_model_checkpoint_callback(
        model_name="flow_matching", dataset_type=cfg.main.dataset.lower(), extra_name=cfg.main.get("extra_name", "default")
    )
    early_stopping_callback = create_early_stopping_callback(patience=50)

    # Initialise MLFlowLogger (wanted to try this for a while so this is me indulging)
    experiment_name = f"sweep_fm_{cfg.main.dataset.lower()}"
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
        logger=mlflow_logger,
        callbacks=[tracker, model_checkpoint_callback],
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

    # Generate samples
    if cfg.main.visualization:
        log.info(f"Loading best model from {model_checkpoint_callback.best_model_path}")
        best_model = FlowMatching.load_from_checkpoint(
            model_checkpoint_callback.best_model_path
        )
        log.info("Generating samples...")
        if (
            cfg.main.dataset.lower() == "two_moons"
            or cfg.main.dataset.lower() == "2d_gaussians"
        ):
            final_samples = best_model.sample(num_samples=2000)

            X = X_train.cpu().numpy()  # Move original data to CPU for plotting
            samples = (
                final_samples.cpu().numpy()
            )  # Move generated samples to CPU for plotting

            save_2d_samples(
                samples, X, tracker, "flow_matching", cfg.main.dataset.lower()
            )
        else:
            final_samples = best_model.sample(num_samples=16)

            # Save generated samples
            save_image_samples(final_samples, "flow_matching", cfg.main.dataset.lower())
            plot_loss_function(tracker, "flow_matching", cfg.main.dataset.lower())
        # Final evaluation
        final_metrics = model.run_final_evaluation(final_samples)

    metrics_history = model.get_metrics_history()
    if cfg.main.visualization:
        fig = plot_evaluation_metrics(
            metrics_history,
            "vector_field",
            save_path=f"flow_matching_{cfg.main.dataset}_metrics.png",
        )
        plt.close()

    # Get summary table
    summary_df = create_metrics_summary_table(metrics_history)
    log.info(summary_df)

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
