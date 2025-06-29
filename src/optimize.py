import logging
import os
import sys
import tempfile
import time
import hydra
import numpy as np
import optuna
import pytorch_lightning as pl
import seaborn as sns
import torch
from optuna.trial import TrialState
from optuna.study import MaxTrialsCallback
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from optuna.storages import RDBStorage

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
        secondary_monitor: str = "eval/precision",
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

        secondary_objective = cfg.main.get("optim_objective", "precision")

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
                secondary_monitor=f"eval/{secondary_objective}",
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
            eval_precision = trainer.callback_metrics.get(
                f"eval/{secondary_objective}", None
            )

            # Fallback to large values if missing or not a number
            if train_loss is None or not hasattr(train_loss, "item"):
                train_loss_value = 1e10
            else:
                train_loss_value = train_loss.item()
            if eval_precision is None or not hasattr(eval_precision, "item"):
                eval_precision_value = (
                    0.0 if secondary_objective == "precision" else 10e10
                )
            else:
                eval_precision_value = eval_precision.item()

            return train_loss_value, eval_precision_value
        except Exception as e:
            log.error(f"Training failed: {e}")
            return 1e10, 0.0 if secondary_objective == "precision" else 10e10

        # Return both objectives for multiobjective optimization
        print(f"Callback metrics: {trainer.callback_metrics}")
        train_loss = trainer.callback_metrics["train_loss"].item()
        eval_precision = trainer.callback_metrics[f"eval/{secondary_objective}"].item()

        return train_loss, eval_precision

    finally:
        # Clean up temporary file
        if os.path.exists(temp_config_path):
            os.unlink(temp_config_path)


def run_optimization():
    """Run the multiobjective hyperparameter optimization"""

    with hydra.initialize(config_path="conf", version_base=None):
        cli_overrides = [arg for arg in sys.argv[1:] if "=" in arg]
        cfg = hydra.compose(config_name="config", overrides=[*cli_overrides])

    secondary_objective = cfg.main.get("optim_objective", "precision")

    max_retries = 10

    # Allow user to select storage backend: "postgresql" or "sqlite"
    storage_backend = cfg.main.get("optuna_storage_backend", "sqlite").lower()
    if storage_backend == "sqlite":
        if cfg.model.generative_model == "flow_matching":
            study_name = f"flowmatching_{secondary_objective}_optuna_study"
            storage_url = cfg.main.get(
                "sqlite_url", "sqlite:///flowmatching_study_db.db"
            )
        else:
            study_name = f"diffusion_{secondary_objective}_optuna_study"
            storage_url = cfg.main.get("sqlite_url", "sqlite:///diffusion_study_db.db")
        # For sqlite, no engine_kwargs needed
        storage = RDBStorage(url=storage_url)
    else:
        if cfg.model.generative_model == "flow_matching":
            study_name = f"flowmatching_{secondary_objective}_optuna_study"
            storage_url = cfg.main.get(
                "postgresql_url", "postgresql://x_sofan@localhost/flowmatching_study_db"
            )
        else:
            study_name = f"diffusion_{secondary_objective}_optuna_study"
            storage_url = cfg.main.get(
                "postgresql_url", "postgresql://x_sofan@localhost/diffusion_study_db"
            )

        for attempt in range(max_retries):
            try:
                storage = RDBStorage(
                    url=storage_url, engine_kwargs={"pool_pre_ping": True}
                )
                break
            except Exception as e:
                print(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    raise

    # Create multiobjective study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        directions=[
            "minimize",
            "maximize" if secondary_objective == "precision" else "minimize",
        ],  # minimize both train_loss and eval/precision
        sampler=optuna.samplers.TPESampler(),  # Changed to TPESampler as it is faster in the current version of Optuna (4.4.0)
        load_if_exists=True,
    )

    # Optimize
    study.optimize(objective, callbacks=[MaxTrialsCallback(100)])

    # Get Pareto front solutions
    pareto_front = study.best_trials

    print(f"Found {len(pareto_front)} Pareto optimal solutions:")
    print("=" * 60)

    for i, trial in enumerate(pareto_front):
        train_loss, eval_precision = trial.values
        print(f"Solution {i + 1}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Eval {secondary_objective}: {eval_precision:.4f}")
        print("  Parameters:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        print("-" * 40)

    # Save detailed results
    df = study.trials_dataframe()
    df.to_csv(f"{study_name}_{secondary_objective}_multiobjective_optuna_results.csv")

    # Create Pareto front visualization
    create_pareto_plot(study, study_name, secondary_objective)

    return study


def create_pareto_plot(study, study_name, secondary_objective):
    """Create a visualization of the Pareto front"""
    try:
        import matplotlib.pyplot as plt

        # Get all trial results
        train_losses = []
        eval_precisions = []
        is_pareto = []

        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                train_loss, eval_precision = trial.values
                train_losses.append(train_loss)
                eval_precisions.append(eval_precision)
                is_pareto.append(trial in study.best_trials)

        # Create plot
        plt.figure(figsize=(10, 6))

        # Plot all trials
        non_pareto_train = [tl for tl, ip in zip(train_losses, is_pareto) if not ip]
        non_pareto_precision = [
            ef for ef, ip in zip(eval_precisions, is_pareto) if not ip
        ]
        non_pareto_train = [x for x in non_pareto_train if x < 1e10]
        non_pareto_precision = [x for x in non_pareto_precision if x < 1e10]
        plt.scatter(
            non_pareto_train,
            non_pareto_precision,
            alpha=0.6,
            color="lightblue",
            label="All trials",
        )

        # Plot Pareto front
        pareto_train = [tl for tl, ip in zip(train_losses, is_pareto) if ip]
        pareto_precision = [ef for ef, ip in zip(eval_precisions, is_pareto) if ip]

        pareto_train = [x for x in pareto_train if x < 1e10]
        pareto_precision = [x for x in pareto_precision if x < 1e10]
        plt.scatter(
            pareto_train,
            pareto_precision,
            color="red",
            s=100,
            label="Pareto front",
            zorder=5,
        )

        plt.xlabel("Train Loss")
        plt.ylabel(f"Eval {secondary_objective}")
        plt.title("Multiobjective Optimization Results")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            f"{study_name}_{secondary_objective}_pareto_front.pdf",
            dpi=300,
            bbox_inches="tight",
            format="pdf",
        )
        plt.show()

    except ImportError:
        print(
            "Matplotlib not available for plotting. Install with: pip install matplotlib"
        )


def select_best_solution(study, train_loss_weight=0.5, precision_weight=0.5):
    """
    Select the best solution from Pareto front using weighted approach

    Args:
        study: Optuna study object
        train_loss_weight: Weight for train loss (0-1)
        precision_weight: Weight for precision score (0-1)
    """
    pareto_trials = study.best_trials

    if not pareto_trials:
        raise ValueError("No Pareto optimal solutions found")

    # Normalize objectives to [0, 1] range for fair weighting
    train_losses = [trial.values[0] for trial in pareto_trials]
    eval_precisions = [trial.values[1] for trial in pareto_trials]

    min_train_loss, max_train_loss = min(train_losses), max(train_losses)
    min_eval_precision, max_eval_precision = min(eval_precisions), max(eval_precisions)

    best_trial = None
    best_score = float("inf")

    for trial in pareto_trials:
        train_loss, eval_precision = trial.values

        # Normalize to [0, 1]
        norm_train_loss = (train_loss - min_train_loss) / (
            max_train_loss - min_train_loss + 1e-8
        )
        # Support both minimization (e.g., loss) and maximization (e.g., precision)
        # For minimization: lower is better, so normalized as (x - min) / (max - min)
        # For maximization: higher is better, so normalized as (max - x) / (max - min)
        # Here, we assume train_loss is to be minimized and eval_precision is to be maximized

        # Normalize eval_precision so that lower is better (for weighted sum minimization)
        if secondary_objective == "precision":
            norm_eval_precision = (max_eval_precision - eval_precision) / (
                max_eval_precision - min_eval_precision + 1e-8
            )
        # Normalize FID the same way as loss
        else:
            norm_eval_precision = (eval_precision - min_eval_precision) / (
                max_eval_precision - min_eval_precision + 1e-8
            )

        # Weighted combination
        combined_score = (
            train_loss_weight * norm_train_loss + precision_weight * norm_eval_precision
        )

        if combined_score < best_score:
            best_score = combined_score
            best_trial = trial

    print(
        f"\nSelected best solution (weights: train_loss={train_loss_weight}, {secondary_objective}={precision_weight}):"
    )
    print(f"  Train Loss: {best_trial.values[0]:.4f}")
    print(f"  Eval {secondary_objective}: {best_trial.values[1]:.4f}")
    print("  Parameters:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    return best_trial


if __name__ == "__main__":
    study = run_optimization()

    # You can select the best solution based on your preferences
    # Option 1: Equal weighting
    best_equal = select_best_solution(
        study, train_loss_weight=0.5, precision_weight=0.5
    )

    # Option 2: Prioritize image quality (precision)
    best_quality = select_best_solution(
        study, train_loss_weight=0.3, precision_weight=0.7
    )

    # Option 3: Prioritize training stability (loss)
    best_stability = select_best_solution(
        study, train_loss_weight=0.7, precision_weight=0.3
    )
