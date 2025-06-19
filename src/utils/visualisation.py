import torchvision
import torch
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk", font="DejaVu Sans")


def save_image_samples(samples: torch.Tensor, model_name: str, dataset: str):
    """Save generated samples as images."""
    # Denormalize from [-1, 1] to [0, 1]
    samples = (samples + 1) / 2
    samples = torch.clamp(samples, 0, 1)

    # Create grid of images
    grid = torchvision.utils.make_grid(samples, nrow=4, padding=2)

    # Save image
    torchvision.utils.save_image(grid, f"{model_name}_{dataset}_final_images.png")


def save_2d_samples(samples, X, tracker, model_name, dataset):
    # Plot results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Get current seaborn palette
    palette = sns.color_palette()
    colour_orig = palette[0]
    colour_gen = palette[1]

    # Original data
    sns.scatterplot(x=X[:, 0], y=X[:, 1], alpha=0.6, s=20, ax=ax1, color=colour_orig)
    ax1.set_title("Original Data")
    ax1.set_xlabel("X₁")
    ax1.set_ylabel("X₂")
    ax1.set_aspect("equal")

    # Generated samples
    sns.scatterplot(
        x=samples[:, 0], y=samples[:, 1], alpha=0.6, s=20, ax=ax2, color=colour_gen
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

    # Comparison
    sns.scatterplot(
        x=X[:, 0],
        y=X[:, 1],
        alpha=0.4,
        s=20,
        label="Original",
        color=colour_orig,
        ax=ax4,
    )
    sns.scatterplot(
        x=samples[:, 0],
        y=samples[:, 1],
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
    plt.savefig(f"{model_name}_{dataset}_results.png", dpi=300)
    plt.show()


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


def plot_loss_function(tracker, model_name, dataset):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        x=range(len(tracker.train_losses)),
        y=tracker.train_losses,
        ax=ax,
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.tight_layout()
    plt.savefig(f"{model_name}_{dataset}_loss_function.png", dpi=300)
    plt.show()
