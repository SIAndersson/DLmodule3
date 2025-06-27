import logging
from pathlib import Path
from typing import Dict, List, Tuple
import pickle

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import torch
from omegaconf import DictConfig, OmegaConf

from utils.seeding import set_seed
from utils.dataset import GenerativeDataModule
from utils.callbacks import (
    create_evaluation_config,
    MetricTracker,
    create_model_checkpoint_callback,
)

from flow_matching import FlowMatching
from diffusion import DiffusionModel


log = logging.getLogger(__name__)


def save_multi_seed_2d_samples(
    all_samples: Dict[int, np.ndarray],
    X: np.ndarray,
    all_trackers: Dict[int, MetricTracker],
    model_name: str,
    dataset: str,
    seeds: List[int],
    extra_name: str,
):
    """
    Plot results for multiple seeds with different colors.

    Args:
        all_samples: Dictionary mapping seed -> generated samples
        X: Original training data
        all_trackers: Dictionary mapping seed -> MetricTracker
        model_name: Name of the model
        dataset: Name of the dataset
        seeds: List of seeds used
        extra_name: suffix to add to file name
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Get a color palette with enough colors for all seeds
    palette = sns.color_palette("husl", n_colors=len(seeds))

    # Original data (consistent across all seeds)
    sns.scatterplot(
        x=X[:, 0], y=X[:, 1], alpha=0.6, s=20, ax=ax1, color="grey", label="Original"
    )
    ax1.set_title("Original Data")
    ax1.set_xlabel("X₁")
    ax1.set_ylabel("X₂")
    ax1.set_aspect("equal")

    # Generated samples for each seed
    for i, seed in enumerate(seeds):
        samples = all_samples[seed]
        color = palette[i]

        sns.scatterplot(
            x=samples[:, 0],
            y=samples[:, 1],
            alpha=0.3,
            s=20,
            ax=ax2,
            color=color,
            label=f"Seed {seed}",
        )

    ax2.set_title("Generated Samples (All Seeds)")
    ax2.set_xlabel("X₁")
    ax2.set_ylabel("X₂")
    ax2.set_aspect("equal")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Training loss for each seed
    for i, seed in enumerate(seeds):
        tracker = all_trackers[seed]
        color = palette[i]

        sns.lineplot(
            x=range(len(tracker.train_losses)),
            y=tracker.train_losses,
            ax=ax3,
            color=color,
            label=f"Seed {seed}",
        )

    ax3.set_title("Training Loss (All Seeds)")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Loss")
    ax3.legend()

    # Comparison plot with original data and all generated samples
    sns.scatterplot(
        x=X[:, 0],
        y=X[:, 1],
        alpha=0.6,
        s=15,
        label="Original",
        color="grey",
        ax=ax4,
    )

    for i, seed in enumerate(seeds):
        samples = all_samples[seed]
        color = palette[i]

        sns.scatterplot(
            x=samples[:, 0],
            y=samples[:, 1],
            alpha=0.3,
            s=15,
            label=f"Generated (Seed {seed})",
            color=color,
            ax=ax4,
        )

    ax4.set_title("Comparison (All Seeds)")
    ax4.set_xlabel("X₁")
    ax4.set_ylabel("X₂")
    ax4.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax4.set_aspect("equal")

    plt.tight_layout()

    # Save the plot
    root_dir = Path(__file__).resolve().parent.parent
    eval_dir = root_dir / "multi_seed_results" / "evaluation_plots"
    eval_dir.mkdir(parents=True, exist_ok=True)
    save_file = eval_dir / f"{model_name}_{dataset}_multi_seed_results_{extra_name}.pdf"
    plt.savefig(save_file, dpi=300, bbox_inches="tight", format='pdf')
    plt.show()

    return fig


def save_seed_comparison_metrics(
    all_metrics: Dict[int, Dict],
    model_name: str,
    dataset: str,
    seeds: List[int],
    extra_name: str,
):
    """
    Plot comparison of key metrics across seeds.

    Args:
        all_metrics: Dictionary mapping seed -> metrics_history
        model_name: Name of the model
        dataset: Name of the dataset
        seeds: List of seeds used
        extra_name: suffix to add to file name
    """
    # Select key metrics to plot
    key_metrics = ["distance_entropy", "wasserstein_distance", "coverage", "precision"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    palette = sns.color_palette("husl", n_colors=len(seeds))

    for idx, metric in enumerate(key_metrics):
        ax = axes[idx]

        for i, seed in enumerate(seeds):
            metrics_history = all_metrics[seed]
            if metric in metrics_history and len(metrics_history[metric]) > 0:
                epochs = metrics_history["epoch"]
                values = metrics_history[metric]
                color = palette[i]

                sns.lineplot(
                    x=epochs, y=values, ax=ax, color=color, label=f"Seed {seed}"
                )

        ax.set_title(f"{metric.replace('_', ' ').title()}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.legend()

    plt.tight_layout()

    # Save the plot
    root_dir = Path(__file__).resolve().parent.parent
    eval_dir = root_dir / "multi_seed_results" / "evaluation_plots"
    eval_dir.mkdir(parents=True, exist_ok=True)
    save_file = eval_dir / f"{model_name}_{dataset}_metrics_comparison_{extra_name}.pdf"
    plt.savefig(save_file, dpi=300, bbox_inches="tight", format='pdf')
    plt.show()

    return fig


def train_single_seed(
    cfg: DictConfig, seed: int, extra_name: str
) -> Tuple[torch.Tensor, Dict, MetricTracker, float, float]:
    """
    Train the model with a single seed and return results.

    Returns:
        samples, metrics_history, tracker, final_train_loss, final_coverage
    """
    log.info(f"Training with seed {seed}")

    # Set seed
    set_seed(seed)

    # Initialize model
    eval_config = create_evaluation_config(
        log,
        cfg,
        model_type="vector_field"
        if cfg.model.generative_model == "flow_matching"
        else "diffusion",
        evaluation_level=cfg.main.evaluation_level,
    )

    print(f"Setting up dataset {cfg.model.generative_model}...")
    if cfg.model.generative_model == "flow_matching":
        model = FlowMatching(cfg.model, eval_config)
    elif cfg.model.generative_model == "diffusion":
        model = DiffusionModel(cfg.model, eval_config)
    else:
        raise NotImplementedError

    # Setup dataset
    if cfg.main.gradient_accumulation:
        gradient_accumulation = cfg.main.batch_size // 32
    else:
        gradient_accumulation = 1

    datamodule = GenerativeDataModule(cfg, log)
    datamodule.prepare_data()
    datamodule.setup(stage="fit")

    # Setup trainer
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

    tracker = MetricTracker()
    model_checkpoint_callback = create_model_checkpoint_callback(
        model_name=f"{cfg.model.generative_model}_seed_{seed}_{extra_name}",
        dataset_type=cfg.main.dataset.lower(),
    )

    trainer = pl.Trainer(
        max_epochs=cfg.main.max_epochs,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        callbacks=[tracker, model_checkpoint_callback],
        enable_progress_bar=True,
        log_every_n_steps=10,
        gradient_clip_val=cfg.main.grad_clip,
        accumulate_grad_batches=gradient_accumulation,
        num_sanity_val_steps=0,
        logger=False,  # Disable logging for cleaner output
    )

    try:
        trainer.fit(model, datamodule)
        log.info(f"Training complete for seed {seed}")

        # Load best model and generate samples
        if cfg.model.generative_model == "flow_matching":
            best_model = FlowMatching.load_from_checkpoint(
                model_checkpoint_callback.best_model_path
            )
            final_samples = model.sample(num_samples=2000)
        elif cfg.model.generative_model == "diffusion":
            device = next(model.parameters()).device
            best_model = DiffusionModel.load_from_checkpoint(
                model_checkpoint_callback.best_model_path
            )
            final_samples = model.sample((2000, 2), device)

        # Generate samples
        metrics_history = model.get_metrics_history()

        # Calculate final metrics
        final_train_loss = tracker.train_losses[-1]
        final_coverage = (
            metrics_history["coverage"][-1] if metrics_history["coverage"] else 0.0
        )

        # Handle inf/NaN values
        if not np.isfinite(final_train_loss) or not np.isfinite(final_coverage):
            last_finite_train_idx = np.where(np.isfinite(tracker.train_losses))[0]
            last_finite_coverage_idx = np.where(
                np.isfinite(metrics_history["coverage"])
            )[0]

            if len(last_finite_train_idx) > 0:
                final_train_loss = tracker.train_losses[last_finite_train_idx[-1]]
            if len(last_finite_coverage_idx) > 0:
                final_coverage = metrics_history["coverage"][
                    last_finite_coverage_idx[-1]
                ]

        return final_samples, metrics_history, tracker, final_train_loss, final_coverage

    except Exception as e:
        log.error(f"Training failed for seed {seed}: {e}")
        return None, {}, None, 1e10, 0.0


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main_multi_seed(cfg: DictConfig):
    """
    Main function to train with multiple seeds and create visualizations.
    """
    # Define seeds to use
    seeds = cfg.main.get(
        "seeds", [666, 123, 42, 10, 7]
    )  # Default seeds, reversing bc 10 is the only one that works for default flow matching so we want it visible in visualization
    extra_name = cfg.main.get("extra_name", "default")

    log.info(f"Training {cfg.model.generative_model} model with seeds: {seeds}")

    # Storage for results
    all_samples = {}
    all_metrics = {}
    all_trackers = {}
    all_train_losses = {}
    all_coverages = {}

    # Train with each seed
    for seed in seeds:
        samples, metrics_history, tracker, train_loss, coverage = train_single_seed(
            cfg, seed, extra_name
        )

        if samples is not None:
            all_samples[seed] = samples.cpu().numpy()
            all_metrics[seed] = metrics_history
            all_trackers[seed] = tracker
            all_train_losses[seed] = train_loss
            all_coverages[seed] = coverage

    # Get original data for visualization
    datamodule = GenerativeDataModule(cfg, log)
    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    X_train = datamodule.get_original_data().cpu().numpy()

    # Create visualizations if we have 2D data
    if (
        cfg.main.dataset.lower() == "two_moons"
        or cfg.main.dataset.lower() == "2d_gaussians"
    ):
        # Create multi-seed visualization
        save_multi_seed_2d_samples(
            all_samples,
            X_train,
            all_trackers,
            cfg.model.generative_model,
            cfg.main.dataset.lower(),
            seeds,
            extra_name,
        )

        # Create metrics comparison
        save_seed_comparison_metrics(
            all_metrics,
            cfg.model.generative_model,
            cfg.main.dataset.lower(),
            seeds,
            extra_name,
        )

    # Save results to file for later analysis
    results = {
        "seeds": seeds,
        "samples": all_samples,
        "metrics": all_metrics,
        "train_losses": all_train_losses,
        "coverages": all_coverages,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }

    root_dir = Path(__file__).resolve().parent.parent
    results_dir = root_dir / "multi_seed_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = (
        results_dir
        / f"{cfg.model.generative_model}_{cfg.main.dataset.lower()}_multi_seed_results_{extra_name}.pkl"
    )
    with open(results_file, "wb") as f:
        pickle.dump(results, f)

    log.info(f"Results saved to {results_file}")

    # Print summary statistics
    log.info("\n" + "=" * 50)
    log.info("MULTI-SEED TRAINING SUMMARY")
    log.info("=" * 50)

    for seed in seeds:
        if seed in all_train_losses:
            log.info(
                f"Seed {seed:3d}: Train Loss = {all_train_losses[seed]:.4f}, "
                f"Coverage = {all_coverages[seed]:.4f}"
            )

    if all_train_losses:
        mean_train_loss = np.mean(list(all_train_losses.values()))
        std_train_loss = np.std(list(all_train_losses.values()))
        mean_coverage = np.mean(list(all_coverages.values()))
        std_coverage = np.std(list(all_coverages.values()))

        log.info(f"\nMean Train Loss: {mean_train_loss:.4f} ± {std_train_loss:.4f}")
        log.info(f"Mean coverage: {mean_coverage:.4f} ± {std_coverage:.4f}")

    return results


if __name__ == "__main__":
    main_multi_seed()
