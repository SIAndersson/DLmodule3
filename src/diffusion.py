import logging

import hydra
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from pytorch_lightning.callbacks import Callback
from sklearn.datasets import make_moons
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from utils.load_huggingface_data import load_huggingface_data
from utils.models import CNN, MLP
from utils.seeding import set_seed

sns.set_theme(style="whitegrid", context="talk", font="DejaVu Sans")

log = logging.getLogger(__name__)


class MetricTracker(Callback):
    def __init__(self):
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get("train_loss")
        if loss is not None:
            self.train_losses.append(loss.item())


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

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(
            -1, 1
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
        x = batch[0]  # Assume batch is (data, labels) or just (data,)
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
        betas_t = self.betas[t].reshape(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(
            -1, 1
        )
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].reshape(-1, 1)

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


def create_2d_dataset(n_samples=10000):
    """
    Create a 2D mixture of Gaussians dataset for visualization
    This creates a simple but interesting distribution to model
    """
    # Create mixture of 4 Gaussians in 2D
    centers = torch.tensor([[-2, -2], [-2, 2], [2, -2], [2, 2]], dtype=torch.float32)
    n_per_cluster = n_samples // 4

    data = []
    for center in centers:
        cluster_data = torch.randn(n_per_cluster, 2) * 0.5 + center
        data.append(cluster_data)

    return torch.cat(data, dim=0)


def create_two_moons_data(n_samples):
    X, _ = make_moons(n_samples=n_samples, noise=0.1, random_state=42)
    return torch.FloatTensor(X)


def create_dataset(cfg: DictConfig):
    """
    Create dataset based on configuration.
    """
    if cfg.main.dataset.lower() == "two_moons":
        log.info("Creating Two Moons dataset...")
        return create_two_moons_data(cfg.main.num_samples)
    elif cfg.main.dataset.lower() == "2d_gaussians":
        log.info("Creating 2D Gaussian mixture dataset...")
        return create_2d_dataset(cfg.main.num_samples)
    elif cfg.main.dataset.lower() == "ffhq":
        dataset_name = "bitmind/ffhq-256"
        log.info(f"Loading dataset: {dataset_name}")
        return load_huggingface_data(dataset_name)
    else:
        raise ValueError(f"Unknown dataset: {cfg.main.dataset}")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """
    Complete training pipeline with visualization
    """
    # Set random seeds for reproducibility
    set_seed(cfg.main.seed)

    # Create dataset
    data = create_dataset(cfg)

    dataset = TensorDataset(data)
    dataloader = DataLoader(
        dataset, batch_size=cfg.main.batch_size, shuffle=True, num_workers=0
    )

    # Initialize model
    log.info("Initializing diffusion model...")
    model = DiffusionModel(cfg.model)

    # Train model
    log.info("Training model...")
    tracker = MetricTracker()
    trainer = pl.Trainer(
        max_epochs=cfg.main.max_epochs,
        accelerator="auto",
        callbacks=[tracker],
        enable_progress_bar=True,
        log_every_n_steps=10,
        gradient_clip_val=cfg.main.grad_clip,
    )
    trainer.fit(model, dataloader)
    log.info("Training complete.")

    # Generate samples
    log.info("Generating samples...")
    model.eval()
    device = next(model.parameters()).device
    generated_samples = model.sample((2000, 2), device)

    # Move to CPU for plotting
    original_data = data.cpu().numpy()
    generated_data = generated_samples.cpu().numpy()

    # Get current seaborn palette
    palette = sns.color_palette()
    colour_orig = palette[0]
    colour_gen = palette[1]

    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Original data
    sns.scatterplot(
        x=original_data[:, 0],
        y=original_data[:, 1],
        alpha=0.6,
        s=20,
        ax=ax1,
        color=colour_orig,
    )
    ax1.set_title("Original Data Distribution")
    ax1.set_xlabel("X₁")
    ax1.set_ylabel("X₂")
    ax1.set_aspect("equal")

    # Generated data
    sns.scatterplot(
        x=generated_data[:, 0],
        y=generated_data[:, 1],
        alpha=0.6,
        s=20,
        ax=ax2,
        color=colour_gen,
    )
    ax2.set_title("Generated Samples")
    ax2.set_xlabel("X₁")
    ax2.set_ylabel("X₂")
    ax2.set_aspect("equal")

    # Training loss
    sns.lineplot(
        x=range(len(tracker.train_losses)),
        y=tracker.train_losses,
        ax=ax3,
    )
    ax3.set_title("Training loss")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Loss")

    # Overlay comparison
    sns.scatterplot(
        x=original_data[:, 0],
        y=original_data[:, 1],
        alpha=0.4,
        s=20,
        label="Original",
        color=colour_orig,
        ax=ax4,
    )
    sns.scatterplot(
        x=generated_data[:, 0],
        y=generated_data[:, 1],
        alpha=0.4,
        s=20,
        label="Generated",
        color=colour_gen,
        ax=ax4,
    )
    ax4.set_title("Comparison")
    ax4.set_xlabel("X₁")
    ax4.set_ylabel("X₂")
    ax4.legend()
    ax4.set_aspect("equal")

    plt.tight_layout()
    plt.savefig("diffusion_training_results.png", dpi=300)
    plt.show()

    # Show diffusion process visualization
    log.info("Visualizing forward diffusion process...")
    visualize_diffusion_process(model, data[:100])

    return model


def visualize_diffusion_process(model, samples):
    """
    Visualize how the forward diffusion process gradually adds noise
    """
    model.eval()
    device = next(model.parameters()).device
    samples = samples.to(device)

    timesteps_to_show = [0, 50, 100, 150, 199]
    fig, axes = plt.subplots(1, len(timesteps_to_show), figsize=(20, 4))

    for i, t in enumerate(timesteps_to_show):
        t_tensor = torch.full((samples.shape[0],), t, device=device)
        noisy_samples = model.q_sample(samples, t_tensor)
        noisy_samples = noisy_samples.cpu().numpy()

        axes[i].scatter(noisy_samples[:, 0], noisy_samples[:, 1], alpha=0.6, s=10)
        axes[i].set_title(f"Timestep t={t}")
        axes[i].set_xlabel("X₁")
        axes[i].set_ylabel("X₂")
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlim(-4, 4)
        axes[i].set_ylim(-4, 4)

    plt.suptitle("Forward Diffusion Process: Gradual Noise Addition")
    plt.tight_layout()
    plt.savefig("diffusion_forward_process.png", dpi=300)
    plt.show()


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
