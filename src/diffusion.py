import math

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback
from sklearn.datasets import make_moons
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from utils.seeding import set_seed

# Set random seed for reproducibility
set_seed(10)


class MetricTracker(Callback):
    def __init__(self):
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get("train_loss")
        if loss is not None:
            self.train_losses.append(loss.item())


class SimpleUNet(nn.Module):
    """
    Simple U-Net architecture for the denoising network ε_θ(x_t, t)

    Mathematical foundation:
    The network learns to predict the noise ε ~ N(0, I) that was added
    at timestep t, enabling reverse diffusion: x_{t-1} = μ_θ(x_t, t) + σ_t z
    """

    def __init__(self, input_dim, hidden_dim=128, time_embed_dim=32, num_layers=4):
        super().__init__()
        self.input_dim = input_dim
        self.time_embed_dim = time_embed_dim

        # Time embedding: converts timestep t to sinusoidal embeddings
        # This helps the network understand which diffusion step we're at
        self.time_embed = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Main network: predicts noise ε given noisy input x_t and timestep t
        layers = [nn.Linear(input_dim + hidden_dim, hidden_dim)]
        for _ in range(num_layers - 1):
            layers.extend([nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)])
        layers.extend([nn.ReLU(), nn.Linear(hidden_dim, input_dim)])

        self.net = nn.Sequential(*layers)

    def get_time_embedding(self, timesteps):
        """
        Sinusoidal time embeddings as used in "Attention Is All You Need"
        Helps the network distinguish between different diffusion timesteps
        """
        device = timesteps.device
        half_dim = self.time_embed_dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings

    def forward(self, x, t):
        """
        Forward pass: ε_θ(x_t, t) -> predicted noise

        Args:
            x: noisy data at timestep t, shape (batch_size, input_dim)
            t: timestep, shape (batch_size,)

        Returns:
            predicted noise ε, shape (batch_size, input_dim)
        """
        t_embed = self.get_time_embedding(t)
        t_embed = self.time_embed(t_embed)

        # Concatenate noisy input with time embedding
        x_with_time = torch.cat([x, t_embed], dim=-1)
        return self.net(x_with_time)


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
        input_dim,
        num_timesteps=1000,
        lr=1e-3,
        hidden_dim=128,
        num_layers=4,
        time_embed_dim=32,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_timesteps = num_timesteps
        self.lr = lr

        # Initialize the denoising network
        self.model = SimpleUNet(input_dim, hidden_dim, time_embed_dim, num_layers)

        # Pre-compute diffusion schedule parameters
        self.register_buffer("betas", self._cosine_beta_schedule(num_timesteps))
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
        return AdamW(self.parameters(), lr=self.lr, weight_decay=1e-3)

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


def train_and_visualize():
    """
    Complete training pipeline with visualization
    """
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Create dataset
    print("Creating 2D mixture of Gaussians dataset...")
    data = create_two_moons_data(8000)
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=0)

    # Initialize model
    print("Initializing diffusion model...")
    model = DiffusionModel(
        input_dim=2,
        num_timesteps=1000,
        lr=2e-3,
        hidden_dim=256,
        num_layers=10,
        time_embed_dim=128,
    )

    # Train model
    print("Training model...")
    tracker = MetricTracker()
    trainer = pl.Trainer(
        max_epochs=500,
        accelerator="auto",
        callbacks=[tracker],
        enable_progress_bar=True,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
    )
    trainer.fit(model, dataloader)

    # Generate samples
    print("Generating samples...")
    model.eval()
    device = next(model.parameters()).device
    generated_samples = model.sample((2000, 2), device)

    # Move to CPU for plotting
    original_data = data.cpu().numpy()
    generated_data = generated_samples.cpu().numpy()

    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Original data
    ax1.scatter(original_data[:, 0], original_data[:, 1], alpha=0.6, s=20)
    ax1.set_title("Original Data Distribution")
    ax1.set_xlabel("X₁")
    ax1.set_ylabel("X₂")
    ax1.set_aspect("equal")

    # Generated data
    ax2.scatter(
        generated_data[:, 0], generated_data[:, 1], alpha=0.6, s=20, color="red"
    )
    ax2.set_title("Generated Samples")
    ax2.set_xlabel("X₁")
    ax2.set_ylabel("X₂")
    ax2.set_aspect("equal")

    # Training loss
    ax3.plot(tracker.train_losses)
    ax3.set_title("Training loss")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Loss")

    # Overlay comparison
    ax4.scatter(
        original_data[:, 0],
        original_data[:, 1],
        alpha=0.4,
        s=20,
        label="Original",
        color="blue",
    )
    ax4.scatter(
        generated_data[:, 0],
        generated_data[:, 1],
        alpha=0.4,
        s=20,
        label="Generated",
        color="red",
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
    print("Visualizing forward diffusion process...")
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
    print("=== Denoising Diffusion Probabilistic Model (DDPM) ===")
    print("\nMathematical Foundation:")
    print("• Forward: q(x_t|x_0) = N(x_t; √(ᾱ_t)x_0, (1-ᾱ_t)I)")
    print("• Reverse: p_θ(x_{t-1}|x_t) = N(x_{t-1}; μ_θ(x_t,t), σ_t²I)")
    print("• Training: L = E[||ε - ε_θ(x_t,t)||²]")
    print("\nStarting training...\n")

    model = train_and_visualize()
    print("\nTraining complete!")
