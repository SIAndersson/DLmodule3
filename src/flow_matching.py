import logging
from typing import Optional

import hydra
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from pytorch_lightning.callbacks import Callback
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from utils.dataset import create_dataset
from utils.models import CNN, MLP
from utils.seeding import set_seed
from utils.visualisation import save_2d_samples, save_image_samples, plot_loss_function

# Configure matplotlib for better aesthetics
sns.set_theme(style="whitegrid", context="talk", font="DejaVu Sans")

log = logging.getLogger(__name__)

if torch.cuda.is_available():
    # Get properties of the first available GPU
    device_props = torch.cuda.get_device_properties(0)
    if device_props.major >= 7:
        torch.set_float32_matmul_precision("high")
        print("Tensor cores enabled globally")


class MetricTracker(Callback):
    def __init__(self):
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get("train_loss")
        if loss is not None:
            self.train_losses.append(loss.item())

    def on_validation_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get("val_loss")
        if loss is not None:
            self.val_losses.append(loss.item())


class FlowMatching(pl.LightningModule):
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

        # Vector field network v_θ(x, t)
        if model_cfg.model_type.upper() == "MLP":
            self.vector_field = MLP(
                input_dim=model_cfg.input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                time_embed_dim=self.time_embed_dim,
                model_type="vector_field",
            )
            self.dim = (model_cfg.input_dim)
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
        x = torch.randn(num_samples, *(self.dim), device=device)

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

    def training_step(self, batch, batch_idx):
        """Training step - compute flow matching loss."""
        x_1 = batch[0] if isinstance(batch, (list, tuple)) else batch
        loss = self.flow_matching_loss(x_1)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step - compute flow matching loss."""
        x_1 = batch[0] if isinstance(batch, (list, tuple)) else batch
        loss = self.flow_matching_loss(x_1)

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Configure Adam optimizer."""
        return AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


# Example usage and testing
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Set random seed for reproducibility
    set_seed(cfg.main.seed)

    log.info("Initializing Flow Matching model...")
    # Initialize model
    model = FlowMatching(cfg.model)

    # Create sample data
    log.info("Setting up dataset...")
    X_train = create_dataset(cfg, log)

    # Define a simple PyTorch Lightning DataLoader
    dataset = TensorDataset(X_train)
    dataloader = DataLoader(
        dataset, batch_size=cfg.main.batch_size, shuffle=True, num_workers=0
    )

    # Initialize PyTorch Lightning Trainer and fit the model
    log.info("Training model...")
    tracker = MetricTracker()
    trainer = pl.Trainer(
        max_epochs=cfg.main.max_epochs,
        callbacks=[tracker],
        accelerator="auto",
        log_every_n_steps=10,
        gradient_clip_val=cfg.main.grad_clip,
    )
    trainer.fit(model, dataloader)
    log.info("Training complete.")

    # Generate samples
    log.info("Generating samples...")

    # TODO: Set up visualisation for image data
    if (
        cfg.main.dataset.lower() == "two_moons"
        or cfg.main.dataset.lower() == "2d_gaussians"
    ):
        samples = model.sample(num_samples=2000)

        X = X_train.cpu().numpy()  # Move original data to CPU for plotting
        samples = samples.cpu().numpy()  # Move generated samples to CPU for plotting

        save_2d_samples(samples, X, tracker, "flow_matching", cfg.main.dataset.lower())
    else:
        final_samples = model.sample(num_samples=16)

        # Save generated samples
        save_image_samples(final_samples, "flow_matching", cfg.main.dataset.lower())
        plot_loss_function(tracker, "flow_matching", cfg.main.dataset.lower())


if __name__ == "__main__":
    main()
