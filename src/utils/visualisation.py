from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchvision

# Set the aesthetic style
sns.set_theme(style="whitegrid", context="talk", font="DejaVu Sans")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 300
#plt.rcParams['svg.fonttype'] = 'none'
#plt.rcParams['pdf.use14corefonts'] = True


def plot_evaluation_metrics(
    metrics_history: Dict,
    model_type: str,
    figsize: Optional[Tuple[int, int]] = None,
    palette: str = "colorblind",
    highlight_color: str = "yellow",
    save_path: Optional[str] = None,
    dpi: int = 300,
    show_best_values: bool = True,
    alpha: float = 0.7,
) -> plt.Figure:
    """
    Create comprehensive visualization of evaluation metrics from FastEvaluationCallback.

    Args:
        callback: FastEvaluationCallback instance with metrics history
        figsize: Figure size (width, height). If None, auto-calculated
        palette: Color palette for lines
        highlight_color: Color for highlighting best metric values
        save_path: Path to save the figure (optional)
        dpi: DPI for saved figure
        show_best_values: Whether to highlight best metric values
        alpha: Transparency for metric lines

    Returns:
        matplotlib Figure object
    """

    # Check if we have data
    if not metrics_history["epoch"]:
        raise ValueError("No metrics data found. Run training with the callback first.")

    # Define metric categories and their properties
    metric_categories = {
        # Lower is better metrics
        "distance_metrics": {
            "metrics": [
                "wasserstein_distance",
                "mmd",
                "js_divergence",
                "energy_distance",
                "spectral_divergence",
                "density_ks_stat",
                "fid",
            ],
            "better": "lower",
            "title": "Distance Metrics (Lower is Better)",
        },
        # Higher is better metrics
        "quality_metrics": {
            "metrics": [
                "coverage",
                "precision",
                "mode_collapse_score",
                "mean_pairwise_distance",
                "min_pairwise_distance",
                "std_pairwise_distance",
                "distance_entropy",
            ],
            "better": "higher",
            "title": "Quality Metrics (Higher is Better)",
        },
        # Special metrics (context dependent)
        "ratio_metrics": {
            "metrics": [
                "duplicate_ratio",
                "log_density_ratio",
                "condition_number_ratio",
            ],
            "better": "zero",  # Generally closer to 0 is better
            "title": "Ratio Metrics (Closer to 0 is Better)",
        },
    }

    # Collect available metrics
    available_metrics = []
    metric_to_category = {}

    for category, info in metric_categories.items():
        for metric in info["metrics"]:
            if metric in metrics_history and len(metrics_history[metric]) > 0:
                available_metrics.append(metric)
                metric_to_category[metric] = category

    if not available_metrics:
        raise ValueError("No valid metrics found in the callback history.")

    # Calculate subplot layout
    n_metrics = len(available_metrics)
    if n_metrics <= 4:
        n_cols = 2
        n_rows = (n_metrics + 1) // 2
    elif n_metrics <= 9:
        n_cols = 3
        n_rows = (n_metrics + 2) // 3
    else:
        n_cols = 4
        n_rows = (n_metrics + 3) // 4

    # Auto-calculate figure size if not provided
    if figsize is None:
        figsize = (n_cols * 5, n_rows * 4)

    # Set style and create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Handle single subplot case
    if n_metrics == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = list(axes) if n_metrics > 1 else [axes]
    else:
        axes = axes.flatten()

    # Get color palette
    colors = sns.color_palette(palette, n_metrics)

    # Plot each metric
    epochs = np.array(metrics_history["epoch"])

    for i, metric in enumerate(available_metrics):
        ax = axes[i]
        values = np.array(metrics_history[metric])

        # Plot the metric
        line = ax.plot(
            epochs,
            values,
            color=colors[i],
            linewidth=2.5,
            alpha=alpha,
            label=metric.replace("_", " ").title(),
        )

        # Highlight best value
        if show_best_values:
            category = metric_to_category[metric]
            better = metric_categories[category]["better"]

            if better == "lower":
                best_idx = np.argmin(values)
                best_label = f"Best: {values[best_idx]:.4f}"
            elif better == "zero":
                best_idx = np.argmin(np.abs(values))
                best_label = f"Best: {values[best_idx]:.4f}"
            else:
                best_idx = np.argmax(values)
                best_label = f"Best: {values[best_idx]:.4f}"

            # Highlight best point
            ax.scatter(
                epochs[best_idx],
                values[best_idx],
                color=highlight_color,
                s=100,
                marker="*",
                zorder=10,
                edgecolors="black",
                linewidth=1,
            )

            # Add text annotation for best value
            ax.annotate(
                best_label,
                xy=(epochs[best_idx], values[best_idx]),
                xytext=(10, 10),
                textcoords="offset points",
                bbox=dict(
                    boxstyle="round,pad=0.3", facecolor=highlight_color, alpha=0.7
                ),
                fontsize=9,
                ha="left",
            )

        # Customize subplot
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=11)
        ax.set_title(
            f"{metric.replace('_', ' ').title()}", fontsize=12, fontweight="bold"
        )
        ax.grid(True, alpha=0.3)

        # Add trend line for better visualization
        if len(epochs) > 2:
            z = np.polyfit(epochs, values, 1)
            p = np.poly1d(z)
            ax.plot(epochs, p(epochs), "--", color="gray", alpha=0.5, linewidth=1)

        # Set axis limits with some padding
        y_min, y_max = values.min(), values.max()
        y_range = y_max - y_min
        if y_range > 0:
            ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

    # Hide unused subplots
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)

    # Add overall title
    fig.suptitle("Evaluation Metrics History", fontsize=16, fontweight="bold", y=0.98)

    model_name = "Flow matching" if model_type == "vector_field" else "diffusion"

    # Add summary statistics as text
    # Add more detailed summary text to the figure
    summary_text = f"""
    Training Summary:
    • Total Epochs: {len(epochs)}
    • Model Type: {model_name.title()}
    """

    fig.text(
        0.02,
        0.02,
        summary_text,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
        verticalalignment="bottom",
    )

    # Adjust layout
    plt.tight_layout(rect=[0, 0.1, 1, 0.96])

    # Save if requested
    if save_path:
        # Determine the root directory (two levels up from this file)
        root_dir = Path(__file__).resolve().parent.parent.parent
        eval_dir = root_dir / "evaluation_plots"
        eval_dir.mkdir(exist_ok=True)
        save_file = eval_dir / save_path
        plt.savefig(save_file, dpi=dpi, bbox_inches="tight", facecolor="white", format='pdf')
        print(f"Figure saved to: {save_file}")

    return fig


def create_metrics_summary_table(metrics_history: Dict) -> pd.DataFrame:
    """
    Create a summary table of final metric values with statistics.

    Args:
        metrics_history: Dict of validation metrics

    Returns:
        pandas DataFrame with metrics summary
    """
    summary_data = []

    for metric, values in metrics_history.items():
        if metric == "epoch" or not values:
            continue

        values_array = np.array(values)

        summary_data.append(
            {
                "Metric": metric.replace("_", " ").title(),
                "Final Value": values_array[-1],
                "Best Value": values_array.min()
                if "distance" in metric
                or "divergence" in metric
                or "ks_stat" in metric
                or "ratio" in metric
                or "mmd" in metric
                or "fid" in metric
                else values_array.max(),
                "Mean": values_array.mean(),
                "Std": values_array.std(),
                "Min": values_array.min(),
                "Max": values_array.max(),
                "Trend": "Improving"
                if values_array[-1] < values_array[0]
                else "Declining"
                if "distance" in metric or "divergence" in metric
                else "Improving"
                if values_array[-1] > values_array[0]
                else "Declining",
            }
        )

    return pd.DataFrame(summary_data)


def save_image_samples(samples: torch.Tensor, model_name: str, dataset: str):
    """Save generated samples as images."""
    # Denormalize from [-1, 1] to [0, 1]
    samples = (samples + 1) / 2
    samples = torch.clamp(samples, 0, 1)

    # Create grid of images
    grid = torchvision.utils.make_grid(samples, nrow=4, padding=2)

    # Save image
    root_dir = Path(__file__).resolve().parent.parent.parent
    eval_dir = root_dir / "evaluation_plots"
    eval_dir.mkdir(exist_ok=True)
    save_file = eval_dir / f"{model_name}_{dataset}_final_images.png"
    torchvision.utils.save_image(grid, save_file)


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

    # Determine the root directory (two levels up from this file)
    root_dir = Path(__file__).resolve().parent.parent.parent
    eval_dir = root_dir / "evaluation_plots"
    eval_dir.mkdir(exist_ok=True)
    save_file = eval_dir / f"{model_name}_{dataset}_results.pdf"

    plt.savefig(save_file, dpi=300, bbox_inches="tight", format='pdf')
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
    root_dir = Path(__file__).resolve().parent.parent.parent
    eval_dir = root_dir / "evaluation_plots"
    eval_dir.mkdir(exist_ok=True)
    save_file = eval_dir / "diffusion_forward_process.pdf"
    plt.savefig(save_file, dpi=300, bbox_inches="tight", format='pdf')
    plt.show()


def plot_loss_function(tracker, model_name, dataset):
    # Get current seaborn palette
    palette = sns.color_palette()
    colour_loss = palette[0]
    colour_fid = palette[1]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot training loss on the left y-axis
    sns.lineplot(
        x=range(len(tracker.train_losses)),
        y=tracker.train_losses,
        label="Training Loss",
        ax=ax1,
        color=colour_loss,
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color=colour_loss)
    ax1.tick_params(axis="y", labelcolor=colour_loss)

    plt.title("Training Loss")
    fig.tight_layout()
    root_dir = Path(__file__).resolve().parent.parent.parent
    eval_dir = root_dir / "evaluation_plots"
    eval_dir.mkdir(exist_ok=True)
    save_file = eval_dir / f"{model_name}_{dataset}_loss_function.pdf"
    plt.savefig(save_file, dpi=300, bbox_inches="tight", format='pdf')
    plt.show()
