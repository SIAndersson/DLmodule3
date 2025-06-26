import mlflow
import numpy as np
import matplotlib.pyplot as plt
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Plot hyperparameter sweep results")
    parser.add_argument(
        "--experiment-name", type=str, required=True, help="Name of the experiment"
    )
    return parser.parse_args()


def is_pareto_efficient(costs):
    """
    Find Pareto-efficient points.
    costs: numpy array of shape (n_points, n_objectives)
    Assumes we want to minimize all objectives.
    Returns a boolean array indicating whether each point is Pareto efficient.
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(
                costs[is_efficient] < c, axis=1
            ) | np.all(costs[is_efficient] == c, axis=1)
            is_efficient[i] = True
    return is_efficient


def get_pareto_front(experiment_name, metric_names):
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="",
        run_view_type=mlflow.entities.ViewType.ALL,
    )
    metrics_list = []
    run_ids = []

    for run in runs:
        try:
            values = [run.data.metrics[m] for m in metric_names]
            values = [float(v) for v in values]
            metrics_list.append(values)
            run_ids.append(run.info.run_id)
        except KeyError:
            continue  # Skip runs missing metrics

    costs = np.array(metrics_list)

    # Invert coverage to turn maximization into minimization
    costs[:, 1] = -costs[:, 1]

    is_efficient = is_pareto_efficient(costs)
    pareto_runs = [
        (run_ids[i], metrics_list[i]) for i in range(len(run_ids)) if is_efficient[i]
    ]
    # Correct coverage back to positive values for display
    pareto_runs = [(rid, [loss, -cov]) for rid, (loss, cov) in pareto_runs]

    return pareto_runs


def plot_pareto_front(pareto_runs):
    losses = [x[1][0] for x in pareto_runs]
    coverage = [x[1][1] for x in pareto_runs]

    plt.figure(figsize=(8, 6))
    plt.scatter(losses, coverage, c="red", label="Pareto Front")
    plt.xlabel("final_train_loss (minimize)")
    plt.ylabel("final_coverage (maximize)")
    plt.title("Pareto Front for sweep_fm_two_moons")
    plt.grid(True)
    plt.legend()
    plt.show()


def get_hyperparameters_of_best_run(
    pareto_runs, experiment_name, metric_name="final_train_loss"
):
    """
    Get hyperparameters of the Pareto front run with the best (lowest) final_train_loss.
    """
    best_run = min(pareto_runs, key=lambda x: x[1][0])  # minimize train loss
    best_run_id = best_run[0]

    client = mlflow.tracking.MlflowClient()
    run = client.get_run(best_run_id)

    print(f"Best run ID: {best_run_id} with {metric_name}: {best_run[1][0]:.4f}")
    print("Hyperparameters:")
    for k, v in run.data.params.items():
        print(f"  {k}: {v}")

    return best_run_id, run.data.params


if __name__ == "__main__":
    args = parse_args()
    experiment_name = args.experiment_name
    metric_names = ["final_train_loss", "final_coverage"]

    pareto_front = get_pareto_front(experiment_name, metric_names)
    if len(pareto_front) == 0:
        print("No runs found with specified metrics.")
    else:
        plot_pareto_front(pareto_front)
        get_hyperparameters_of_best_run(pareto_front, experiment_name)
