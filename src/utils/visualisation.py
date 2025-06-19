import torchvision
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict
import json

# Set the aesthetic style
sns.set_theme(style="whitegrid", context="talk", font="DejaVu Sans")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 300


class EvaluationVisualizer:
    """
    Comprehensive visualization class for generative model evaluation metrics.
    Creates publication-ready plots using seaborn.
    """

    def __init__(self, save_dir: str = "evaluation_plots"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

        # Color palette for different models
        self.model_colors = {
            "diffusion": "#2E86AB",  # Blue
            "vector_field": "#A23B72",  # Purple
            "baseline": "#F18F01",  # Orange
            "target": "#C73E1D",  # Red
        }

        self.model_names = {
            "diffusion": "diffusion",
            "vector_field": "flow_matching",
        }

        # Metric configurations
        self.metric_configs = {
            "wasserstein_dist": {
                "title": "Wasserstein Distance",
                "ylabel": "Distance",
                "lower_is_better": True,
                "format": ".4f",
            },
            "coverage": {
                "title": "Coverage Score",
                "ylabel": "Coverage",
                "lower_is_better": False,
                "format": ".3f",
            },
            "precision": {
                "title": "Precision Score",
                "ylabel": "Precision",
                "lower_is_better": False,
                "format": ".3f",
            },
            "mmd": {
                "title": "Maximum Mean Discrepancy",
                "ylabel": "MMD",
                "lower_is_better": True,
                "format": ".4f",
            },
            "inception_score": {
                "title": "Inception Score",
                "ylabel": "IS",
                "lower_is_better": False,
                "format": ".2f",
            },
            "kid_score": {
                "title": "Kernel Inception Distance",
                "ylabel": "KID",
                "lower_is_better": True,
                "format": ".4f",
            },
        }

    def plot_training_curves(self, metrics_history: Dict, model_name: str = "model"):
        """
        Plot training curves for all metrics over epochs.

        Args:
            metrics_history: Dictionary containing metric histories
            model_name: Name of the model for labeling
        """
        # Filter out metrics that have data
        available_metrics = [
            metric
            for metric in self.metric_configs.keys()
            if metric in metrics_history and len(metrics_history[metric]) > 0
        ]

        if not available_metrics:
            print("No metrics data available for plotting")
            return

        # Calculate subplot grid
        n_metrics = len(available_metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1 and n_cols > 1:
            axes = axes.flatten()
        elif n_rows > 1:
            axes = axes.flatten()

        color = self.model_colors.get(model_name.lower(), "#2E86AB")

        for i, metric in enumerate(available_metrics):
            ax = axes[i] if n_metrics > 1 else axes[0]

            epochs = metrics_history["epoch"]
            values = metrics_history[metric]

            # Create smooth line plot
            sns.lineplot(
                x=epochs,
                y=values,
                ax=ax,
                color=color,
                linewidth=2.5,
                marker="o",
                markersize=4,
                label=self.model_names[model_name].replace("_", " ").title(),
            )

            config = self.metric_configs[metric]
            ax.set_title(config["title"], fontsize=14, fontweight="bold")
            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel(config["ylabel"], fontsize=12)

            # Add value annotations for key points
            if len(values) > 1:
                # Annotate best value
                if config["lower_is_better"]:
                    best_idx = np.argmin(values)
                    best_text = f"Best: {values[best_idx]:{config['format']}}"
                else:
                    best_idx = np.argmax(values)
                    best_text = f"Best: {values[best_idx]:{config['format']}}"

                ax.annotate(
                    best_text,
                    xy=(epochs[best_idx], values[best_idx]),
                    xytext=(10, 10),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
                    fontsize=10,
                )

            # Format y-axis
            ax.grid(True, alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # Remove empty subplots
        for i in range(n_metrics, len(axes)):
            if i < len(axes):
                fig.delaxes(axes[i])

        plt.tight_layout()

        # Save plot
        filename = f"{self.model_names[model_name]}_training_curves.png"
        plt.savefig(self.save_dir / filename, bbox_inches="tight", facecolor="white")
        plt.show()

        print(f"Training curves saved to {self.save_dir / filename}")

    def compare_models(
        self, model_histories: Dict[str, Dict], title: str = "Model Comparison"
    ):
        """
        Compare multiple models side by side.

        Args:
            model_histories: Dict mapping model names to their metric histories
            title: Overall title for the comparison
        """
        # Find common metrics across all models
        all_metrics = set()
        for history in model_histories.values():
            all_metrics.update(history.keys())

        common_metrics = [
            metric
            for metric in self.metric_configs.keys()
            if all(
                metric in history and len(history[metric]) > 0
                for history in model_histories.values()
            )
        ]

        if not common_metrics:
            print("No common metrics found across models")
            return

        # Calculate subplot grid
        n_metrics = len(common_metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1 and n_cols > 1:
            axes = axes.flatten()
        elif n_rows > 1:
            axes = axes.flatten()

        for i, metric in enumerate(common_metrics):
            ax = axes[i] if n_metrics > 1 else axes[0]

            # Plot each model
            for model_name, history in model_histories.items():
                epochs = history["epoch"]
                values = history[metric]
                color = self.model_colors.get(
                    model_name.lower(),
                    np.random.choice(list(self.model_colors.values())),
                )

                sns.lineplot(
                    x=epochs,
                    y=values,
                    ax=ax,
                    color=color,
                    linewidth=2.5,
                    marker="o",
                    markersize=4,
                    label=model_name.replace("_", " ").title(),
                )

            config = self.metric_configs[metric]
            ax.set_title(config["title"], fontsize=14, fontweight="bold")
            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel(config["ylabel"], fontsize=12)
            ax.legend(frameon=True, fancybox=True, shadow=True)
            ax.grid(True, alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # Remove empty subplots
        for i in range(n_metrics, len(axes)):
            if i < len(axes):
                fig.delaxes(axes[i])

        plt.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
        plt.tight_layout()

        # Save plot
        filename = "model_comparison.png"
        plt.savefig(self.save_dir / filename, bbox_inches="tight", facecolor="white")
        plt.show()

        print(f"Model comparison saved to {self.save_dir / filename}")

    def plot_final_metrics_comparison(self, model_histories: Dict[str, Dict]):
        """
        Create a bar plot comparing final metric values across models.
        """
        # Prepare data for plotting
        comparison_data = []

        for model_name, history in model_histories.items():
            if not history["epoch"]:
                continue

            for metric in self.metric_configs.keys():
                if metric in history and history[metric]:
                    final_value = history[metric][-1]
                    comparison_data.append(
                        {
                            "Model": model_name.replace("_", " ").title(),
                            "Metric": self.metric_configs[metric]["title"],
                            "Value": final_value,
                            "Lower_is_Better": self.metric_configs[metric][
                                "lower_is_better"
                            ],
                        }
                    )

        if not comparison_data:
            print("No data available for final metrics comparison")
            return

        df = pd.DataFrame(comparison_data)

        # Get unique metrics for subplots
        unique_metrics = df["Metric"].unique()
        n_metrics = len(unique_metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1 and n_cols > 1:
            axes = axes.flatten()
        elif n_rows > 1:
            axes = axes.flatten()

        for i, metric in enumerate(unique_metrics):
            ax = axes[i] if n_metrics > 1 else axes[0]

            metric_data = df[df["Metric"] == metric]

            # Create bar plot
            sns.barplot(
                data=metric_data,
                x="Model",
                y="Value",
                ax=ax,
                palette=[
                    self.model_colors.get(model.lower().replace(" ", "_"), "#2E86AB")
                    for model in metric_data["Model"]
                ],
                alpha=0.8,
            )

            # Add value labels on bars
            for j, (_, row) in enumerate(metric_data.iterrows()):
                height = row["Value"]
                format_str = self.metric_configs.get(
                    next(
                        k
                        for k, v in self.metric_configs.items()
                        if v["title"] == metric
                    ),
                    {},
                ).get("format", ".3f")

                ax.text(
                    j,
                    height + height * 0.01,
                    f"{height:{format_str}}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

            ax.set_title(metric, fontsize=14, fontweight="bold")
            ax.set_xlabel("Model", fontsize=12)
            ax.set_ylabel("Value", fontsize=12)
            ax.tick_params(axis="x", rotation=45)

            # Highlight the best performing model
            lower_is_better = metric_data.iloc[0]["Lower_is_Better"]
            if lower_is_better:
                best_idx = metric_data["Value"].idxmin()
            else:
                best_idx = metric_data["Value"].idxmax()

            best_model_idx = metric_data.index.get_loc(best_idx)

            # Add a star to the best performing bar
            bars = ax.patches
            if best_model_idx < len(bars):
                bars[best_model_idx].set_edgecolor("gold")
                bars[best_model_idx].set_linewidth(3)

        # Remove empty subplots
        for i in range(n_metrics, len(axes)):
            if i < len(axes):
                fig.delaxes(axes[i])

        plt.suptitle("Final Metrics Comparison", fontsize=16, fontweight="bold", y=0.98)
        plt.tight_layout()

        # Save plot
        filename = "final_metrics_comparison.png"
        plt.savefig(self.save_dir / filename, bbox_inches="tight", facecolor="white")
        plt.show()

        print(f"Final metrics comparison saved to {self.save_dir / filename}")

    def plot_metric_evolution_heatmap(self, model_histories: Dict[str, Dict]):
        """
        Create a heatmap showing how metrics evolve over time for different models.
        """
        # Prepare data
        heatmap_data = []

        for model_name, history in model_histories.items():
            if not history["epoch"]:
                continue

            epochs = history["epoch"]
            for metric in self.metric_configs.keys():
                if metric in history and history[metric]:
                    values = history[metric]

                    # Normalize values (0-1 scale)
                    if self.metric_configs[metric]["lower_is_better"]:
                        # For lower-is-better metrics, invert so higher values are better
                        normalized = 1 - (np.array(values) - np.min(values)) / (
                            np.max(values) - np.min(values) + 1e-8
                        )
                    else:
                        normalized = (np.array(values) - np.min(values)) / (
                            np.max(values) - np.min(values) + 1e-8
                        )

                    for epoch, norm_val in zip(epochs, normalized):
                        heatmap_data.append(
                            {
                                "Model": model_name.replace("_", " ").title(),
                                "Metric": self.metric_configs[metric]["title"],
                                "Epoch": epoch,
                                "Normalized_Score": norm_val,
                            }
                        )

        if not heatmap_data:
            print("No data available for heatmap")
            return

        df = pd.DataFrame(heatmap_data)

        # Create pivot table for heatmap
        pivot_df = df.pivot_table(
            index=["Model", "Metric"],
            columns="Epoch",
            values="Normalized_Score",
            fill_value=0,
        )

        # Create the heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            pivot_df,
            annot=False,
            cmap="RdYlGn",  # Red-Yellow-Green colormap
            cbar_kws={"label": "Normalized Performance (0=worst, 1=best)"},
            linewidths=0.5,
        )

        plt.title("Metric Evolution Heatmap", fontsize=16, fontweight="bold")
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Model - Metric", fontsize=12)
        plt.tight_layout()

        # Save plot
        filename = "metric_evolution_heatmap.png"
        plt.savefig(self.save_dir / filename, bbox_inches="tight", facecolor="white")
        plt.show()

        print(f"Metric evolution heatmap saved to {self.save_dir / filename}")

    def save_metrics_summary(self, model_histories: Dict[str, Dict]):
        """Save a comprehensive metrics summary as JSON and CSV."""

        summary_data = {}
        csv_data = []

        for model_name, history in model_histories.items():
            if not history["epoch"]:
                continue

            model_summary = {
                "total_epochs": len(history["epoch"]),
                "final_metrics": {},
                "best_metrics": {},
                "improvement": {},
            }

            for metric in self.metric_configs.keys():
                if metric in history and history[metric]:
                    values = history[metric]
                    epochs = history["epoch"]

                    final_value = values[-1]
                    initial_value = values[0] if len(values) > 1 else final_value

                    if self.metric_configs[metric]["lower_is_better"]:
                        best_value = min(values)
                        best_epoch = epochs[np.argmin(values)]
                        improvement = (
                            (initial_value - final_value) / initial_value
                        ) * 100
                    else:
                        best_value = max(values)
                        best_epoch = epochs[np.argmax(values)]
                        improvement = (
                            (final_value - initial_value) / initial_value
                        ) * 100

                    model_summary["final_metrics"][metric] = final_value
                    model_summary["best_metrics"][metric] = {
                        "value": best_value,
                        "epoch": best_epoch,
                    }
                    model_summary["improvement"][metric] = improvement

                    # Add to CSV data
                    csv_data.append(
                        {
                            "Model": model_name,
                            "Metric": metric,
                            "Final_Value": final_value,
                            "Best_Value": best_value,
                            "Best_Epoch": best_epoch,
                            "Improvement_Percent": improvement,
                        }
                    )

            summary_data[model_name] = model_summary

        # Save JSON summary
        json_file = self.save_dir / "metrics_summary.json"
        with open(json_file, "w") as f:
            json.dump(summary_data, f, indent=2, default=str)

        # Save CSV summary
        if csv_data:
            csv_df = pd.DataFrame(csv_data)
            csv_file = self.save_dir / "metrics_summary.csv"
            csv_df.to_csv(csv_file, index=False)

            print(f"Metrics summary saved to {json_file} and {csv_file}")

        return summary_data

    def create_comprehensive_report(self, model_histories: Dict[str, Dict]):
        """Create all visualizations and summaries in one go."""
        print("Creating comprehensive evaluation report...")

        # Individual model training curves
        for model_name, history in model_histories.items():
            if history["epoch"]:
                self.plot_training_curves(history, model_name)

        # Model comparison plots
        if len(model_histories) > 1:
            self.compare_models(model_histories)
            self.plot_final_metrics_comparison(model_histories)
            self.plot_metric_evolution_heatmap(model_histories)

        # Save summary
        self.save_metrics_summary(model_histories)

        print(f"Comprehensive report saved to {self.save_dir}")


# Example usage function to add to your callback
def visualize_evaluation_results(eval_callback, model_name: str = "diffusion"):
    """
    Convenience function to create visualizations from evaluation callback.

    Args:
        eval_callback: Instance of FastEvaluationCallback
        model_name: Name of the model for labeling
    """
    visualizer = EvaluationVisualizer()
    metrics_history = eval_callback.get_metrics_history()

    if metrics_history["epoch"]:
        visualizer.plot_training_curves(metrics_history, model_name)
        print("Evaluation visualizations created successfully!")
    else:
        print("No evaluation data available for visualization")


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

    if len(tracker.fid_scores) > 0:
        # Create a second y-axis for FID scores
        ax2 = ax1.twinx()
        sns.lineplot(
            x=range(len(tracker.fid_scores)),
            y=tracker.fid_scores,
            label="FID Score",
            ax=ax2,
            color=colour_fid,
        )
        ax2.set_ylabel("FID Score", color=colour_fid)
        ax2.tick_params(axis="y", labelcolor=colour_fid)

    plt.title("Training Loss{' and FID Score' if len(tracker.fid_scores) > 0 else ''}")
    fig.tight_layout()
    plt.savefig(f"{model_name}_{dataset}_loss_and_fid_function.png", dpi=300)
    plt.show()
