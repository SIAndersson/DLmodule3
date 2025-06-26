import logging
import os
import sys
import tempfile

import hydra
import numpy as np
import optuna
import pytorch_lightning as pl
import seaborn as sns
import torch
from optuna.trial import TrialState
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint

from diffusion import DiffusionModel
from flow_matching import FlowMatching
from utils.callbacks import (
    create_evaluation_config,
)
from utils.dataset import GenerativeDataModule
from utils.seeding import set_seed

sns.set_theme(style="whitegrid", context="talk", font="DejaVu Sans")


log = logging.getLogger(__name__)


class MultiObjectiveMedianPrunerCallback(Callback):
    """Custom median pruner for multi-objective optimization"""

    def __init__(
        self,
        trial: optuna.Trial,
        primary_monitor: str = "train_loss",
        secondary_monitor: str = "eval/fid",
        n_startup_trials: int = 10,
        n_warmup_steps: int = 10,
        interval_steps: int = 5,
    ):
        super().__init__()
        self.trial = trial
        self.primary_monitor = primary_monitor
        self.secondary_monitor = secondary_monitor
        self.n_startup_trials = n_startup_trials
        self.n_warmup_steps = n_warmup_steps
        self.interval_steps = interval_steps
        self.primary_values = {}  # {epoch: value}

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        metrics = trainer.callback_metrics

        # Extract metrics
        primary_score = metrics.get(self.primary_monitor)
        secondary_score = metrics.get(self.secondary_monitor)

        # Store metrics as user attributes
        if primary_score is not None:
            primary_score = float(primary_score)
            self.trial.set_user_attr(f"primary_epoch_{epoch}", primary_score)
            self.primary_values[epoch] = primary_score

        if secondary_score is not None:
            self.trial.set_user_attr(f"secondary_epoch_{epoch}", float(secondary_score))

        # Pruning logic (only for primary objective)
        if primary_score is None:
            return

        # Skip pruning during warmup periods
        if epoch < self.n_warmup_steps:
            return

        # Skip pruning if not enough completed trials
        study = self.trial.study
        completed_trials = study.get_trials(
            deepcopy=False, states=[TrialState.COMPLETE, TrialState.PRUNED]
        )
        if len(completed_trials) < self.n_startup_trials:
            return

        # Apply interval checking
        if (epoch - self.n_warmup_steps) % self.interval_steps != 0 or (
            epoch - self.n_warmup_steps
        ) < 0:
            return

        # Calculate median of best values at this epoch across completed trials
        values_at_epoch = []
        for t in completed_trials:
            attr = f"primary_epoch_{epoch}"
            if attr in t.user_attrs:
                values_at_epoch.append(t.user_attrs[attr])

        if not values_at_epoch:
            return

        median_value = np.median(values_at_epoch)

        # Prune if current value is worse than median (for minimization)
        if primary_score > median_value:
            raise optuna.TrialPruned(
                f"Trial pruned at epoch {epoch}: "
                f"{primary_score:.4f} > median {median_value:.4f} "
                f"(across {len(values_at_epoch)} trials)"
            )


def objective(trial):
    """Multiobjective optimization function"""
    # Sample hyperparameters
    suggested_params = {
        "model.lr": trial.suggest_float("model.lr", 1e-5, 1e-2, log=True),
        "model.weight_decay": trial.suggest_float(
            "model.weight_decay", 1e-4, 1e-2, log=True
        ),
        "main.batch_size": trial.suggest_categorical(
            "main.batch_size", [128, 256, 512]
        ),
        "model.hidden_dim": trial.suggest_categorical(
            "model.hidden_dim", [64, 128, 256, 512]
        ),
        "model.num_layers": trial.suggest_categorical(
            "model.num_layers", [2, 4, 6, 8, 10, 12, 15]
        ),
    }

    # Create temporary config override
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        # Write overrides to temporary file
        overrides = []
        for key, value in suggested_params.items():
            overrides.append(f"{key}={value}")

        temp_config_path = f.name

    try:
        # Initialize Hydra with overrides
        with hydra.initialize(config_path="conf", version_base=None):
            cli_overrides = [arg for arg in sys.argv[1:] if "=" in arg]
            cfg = hydra.compose(
                config_name="config", overrides=[*overrides, *cli_overrides]
            )

        set_seed(cfg.main.seed)

        print(f"Initializing {cfg.model.generative_model} model...")
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

        # Add multiobjective pruning callback
        callbacks = [
            MultiObjectiveMedianPrunerCallback(
                trial,
                primary_monitor="train_loss",
                secondary_monitor="eval/fid",
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=1,
            ),
            EarlyStopping(monitor="train_loss", patience=20),
            ModelCheckpoint(monitor="train_loss", mode="min"),
        ]

        if cfg.main.gradient_accumulation:
            gradient_accumulation = cfg.main.batch_size // 32
        else:
            gradient_accumulation = 1

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

        print(f"Setting up dataset {cfg.main.dataset.lower()}...")
        datamodule = GenerativeDataModule(cfg, log)
        datamodule.prepare_data()
        datamodule.setup(stage="fit")

        trainer = pl.Trainer(
            max_epochs=cfg.main.max_epochs,
            accelerator=accelerator,
            devices=devices,
            strategy=strategy,
            callbacks=callbacks,
            enable_progress_bar=True,  # Better tracking
            logger=False,  # Disable logging for trials
            log_every_n_steps=10,
            gradient_clip_val=cfg.main.grad_clip,
            accumulate_grad_batches=gradient_accumulation,
            num_sanity_val_steps=0,
        )

        # Train model
        try:
            trainer.fit(model, datamodule)
            log.info("Training complete.")
        # Most likely fails due to Cuda OOM, return high values for loss and metric
        except optuna.TrialPruned as e:
            log.error(f"Trial pruned: {e}")
            train_loss = trainer.callback_metrics.get("train_loss", None)
            eval_fid = trainer.callback_metrics.get("eval/fid", None)

            # Fallback to large values if missing or not a number
            if train_loss is None or not hasattr(train_loss, "item"):
                train_loss_value = 1e10
            else:
                train_loss_value = train_loss.item()
            if eval_fid is None or not hasattr(eval_fid, "item"):
                eval_fid_value = 1e10
            else:
                eval_fid_value = eval_fid.item()

            return train_loss_value, eval_fid_value
        except Exception as e:
            log.error(f"Training failed: {e}")
            return 1e10, 1e10

        # Return both objectives for multiobjective optimization
        print(f"Callback metrics: {trainer.callback_metrics}")
        train_loss = trainer.callback_metrics["train_loss"].item()
        eval_fid = trainer.callback_metrics["eval/fid"].item()

        return train_loss, eval_fid

    finally:
        # Clean up temporary file
        if os.path.exists(temp_config_path):
            os.unlink(temp_config_path)


def run_optimization():
    """Run the multiobjective hyperparameter optimization"""
    # Create multiobjective study
    study = optuna.create_study(
        directions=["minimize", "minimize"],  # minimize both train_loss and eval_fid
        sampler=optuna.samplers.NSGAIISampler(seed=10),  # Good for multiobjective
    )

    # Optimize
    study.optimize(objective, n_trials=100)

    # Get Pareto front solutions
    pareto_front = study.best_trials

    print(f"Found {len(pareto_front)} Pareto optimal solutions:")
    print("=" * 60)

    for i, trial in enumerate(pareto_front):
        train_loss, eval_fid = trial.values
        print(f"Solution {i + 1}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Eval FID: {eval_fid:.4f}")
        print("  Parameters:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        print("-" * 40)

    # Save detailed results
    df = study.trials_dataframe()
    df.to_csv("multiobjective_optuna_results.csv")

    # Create Pareto front visualization
    create_pareto_plot(study)

    return study


def create_pareto_plot(study):
    """Create a visualization of the Pareto front"""
    try:
        import matplotlib.pyplot as plt

        # Get all trial results
        train_losses = []
        eval_fids = []
        is_pareto = []

        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                train_loss, eval_fid = trial.values
                train_losses.append(train_loss)
                eval_fids.append(eval_fid)
                is_pareto.append(trial in study.best_trials)

        # Create plot
        plt.figure(figsize=(10, 6))

        # Plot all trials
        non_pareto_train = [tl for tl, ip in zip(train_losses, is_pareto) if not ip]
        non_pareto_fid = [ef for ef, ip in zip(eval_fids, is_pareto) if not ip]
        plt.scatter(
            non_pareto_train,
            non_pareto_fid,
            alpha=0.6,
            color="lightblue",
            label="All trials",
        )

        # Plot Pareto front
        pareto_train = [tl for tl, ip in zip(train_losses, is_pareto) if ip]
        pareto_fid = [ef for ef, ip in zip(eval_fids, is_pareto) if ip]
        plt.scatter(
            pareto_train, pareto_fid, color="red", s=100, label="Pareto front", zorder=5
        )

        plt.xlabel("Train Loss")
        plt.ylabel("Eval FID")
        plt.title("Multiobjective Optimization Results")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("pareto_front.png", dpi=300, bbox_inches="tight")
        plt.show()

    except ImportError:
        print(
            "Matplotlib not available for plotting. Install with: pip install matplotlib"
        )


def select_best_solution(study, train_loss_weight=0.5, fid_weight=0.5):
    """
    Select the best solution from Pareto front using weighted approach

    Args:
        study: Optuna study object
        train_loss_weight: Weight for train loss (0-1)
        fid_weight: Weight for FID score (0-1)
    """
    pareto_trials = study.best_trials

    if not pareto_trials:
        raise ValueError("No Pareto optimal solutions found")

    # Normalize objectives to [0, 1] range for fair weighting
    train_losses = [trial.values[0] for trial in pareto_trials]
    eval_fids = [trial.values[1] for trial in pareto_trials]

    min_train_loss, max_train_loss = min(train_losses), max(train_losses)
    min_eval_fid, max_eval_fid = min(eval_fids), max(eval_fids)

    best_trial = None
    best_score = float("inf")

    for trial in pareto_trials:
        train_loss, eval_fid = trial.values

        # Normalize to [0, 1]
        norm_train_loss = (train_loss - min_train_loss) / (
            max_train_loss - min_train_loss + 1e-8
        )
        norm_eval_fid = (eval_fid - min_eval_fid) / (max_eval_fid - min_eval_fid + 1e-8)

        # Weighted combination
        combined_score = (
            train_loss_weight * norm_train_loss + fid_weight * norm_eval_fid
        )

        if combined_score < best_score:
            best_score = combined_score
            best_trial = trial

    print(
        f"\nSelected best solution (weights: train_loss={train_loss_weight}, fid={fid_weight}):"
    )
    print(f"  Train Loss: {best_trial.values[0]:.4f}")
    print(f"  Eval FID: {best_trial.values[1]:.4f}")
    print("  Parameters:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    return best_trial


if __name__ == "__main__":
    study = run_optimization()

    # You can select the best solution based on your preferences
    # Option 1: Equal weighting
    best_equal = select_best_solution(study, train_loss_weight=0.5, fid_weight=0.5)

    # Option 2: Prioritize image quality (FID)
    best_quality = select_best_solution(study, train_loss_weight=0.3, fid_weight=0.7)

    # Option 3: Prioritize training stability (loss)
    best_stability = select_best_solution(study, train_loss_weight=0.7, fid_weight=0.3)
