import logging
import os
import fnmatch


from itertools import combinations
import pandas as pd
import sys
import colorlog
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import torch

from utils.evaluator import StandaloneGenerativeModelEvaluator
from utils.dataset import load_huggingface_data
from utils.visualisation import (
    save_image_samples,
)

from diffusion import DiffusionModel
from flow_matching import FlowMatching

sns.set_theme(style="whitegrid", context="talk", font="DejaVu Sans")

log = logging.getLogger(__name__)
stdout = colorlog.StreamHandler(stream=sys.stdout)
fmt = colorlog.ColoredFormatter(
    "%(name)s: %(white)s%(asctime)s%(reset)s | %(log_color)s%(levelname)s%(reset)s | %(blue)s%(filename)s:%(lineno)s%(reset)s | %(process)d >>> %(log_color)s%(message)s%(reset)s"
)
stdout.setFormatter(fmt)
log.addHandler(stdout)
log.setLevel(logging.INFO)

# Metric categories with optimization directions
METRIC_CATEGORIES = {
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

if torch.cuda.is_available():
    # Get properties of the first available GPU
    device_props = torch.cuda.get_device_properties(0)
    if device_props.major >= 7:
        torch.set_float32_matmul_precision("high")
        log.info("Tensor cores enabled globally")


def get_metric_direction(metric):
    """Get the optimization direction for a metric."""
    for category in METRIC_CATEGORIES.values():
        if metric in category["metrics"]:
            return category["better"]
    return "higher"  # Default assumption


def normalize_metric_for_ranking(df, metric):
    """Normalize metric values for ranking (higher = better after normalization)."""
    direction = get_metric_direction(metric)
    values = df[metric].copy()

    if direction == "lower":
        # Invert so higher normalized values are better
        max_val = values.max()
        return max_val - values
    elif direction == "zero":
        # Distance from zero, then invert
        abs_values = np.abs(values)
        max_abs = abs_values.max()
        return max_abs - abs_values
    else:  # "higher"
        return values


def get_best_model_for_metric(df, metric):
    """Get the best performing model for a specific metric."""
    direction = get_metric_direction(metric)

    if direction == "lower":
        try:
            return df.loc[df[metric].idxmin(), "model_category"]
        except Exception:
            return df.loc[df[metric].idxmin(), "model"]
    elif direction == "zero":
        abs_diff = np.abs(df[metric])
        try:
            return df.loc[abs_diff.idxmin(), "model_category"]
        except Exception:
            return df.loc[abs_diff.idxmin(), "model"]
    else:  # "higher"
        try:
            return df.loc[df[metric].idxmax(), "model_category"]
        except Exception:
            return df.loc[df[metric].idxmax(), "model"]


def get_performance_score(df, metric, model_idx):
    """Get a normalized performance score (0-1, where 1 is best)."""
    direction = get_metric_direction(metric)
    values = df[metric]

    if direction == "lower":
        # For lower is better: (max - value) / (max - min)
        return (values.max() - values.iloc[model_idx]) / (values.max() - values.min())
    elif direction == "zero":
        # For zero is better: 1 - (abs(value) / max_abs)
        abs_values = np.abs(values)
        if abs_values.max() == 0:
            return 0
        return 1 - (abs_values.iloc[model_idx] / abs_values.max())
    else:  # "higher"
        # For higher is better: (value - min) / (max - min)
        try:
            return (values.iloc[model_idx] - values.min()) / (
                values.max() - values.min()
            )
        except Exception as e:
            log.critical(
                f"Performance calculation failed. Values are {values} and metric is {metric}."
            )
            raise e


def plot_overview(df, save_path, figsize=(24, 18)):
    """
    Create a comprehensive overview of all metrics with multiple visualizations.

    Args:
        df: DataFrame with columns ['model', 'type', 'optimized', ...metrics...]
        save_path: Where to save figure
        figsize: Figure size tuple
    """
    # Get metric columns (exclude model, type, optimized)
    metric_cols = [
        col for col in df.columns if col not in ["model", "type", "optimized"]
    ]

    # Create figure with more spacing
    fig = plt.figure(figsize=figsize)

    # Adjust spacing between subplots
    plt.subplots_adjust(
        left=0.08,  # left margin
        bottom=0.08,  # bottom margin
        right=0.95,  # right margin
        top=0.93,  # top margin
        wspace=0.35,  # width spacing between subplots
        hspace=0.45,  # height spacing between subplots
    )

    # 1. Correlation heatmap of all metrics
    plt.subplot(2, 4, 1)
    corr_matrix = df[metric_cols].corr()

    # Truncate metric names for correlation matrix
    short_names = [
        name[:12] + "..." if len(name) > 15 else name for name in metric_cols
    ]
    corr_matrix.index = short_names
    corr_matrix.columns = short_names

    ax = sns.heatmap(
        corr_matrix,
        annot=False,
        cmap="coolwarm",
        center=0,
        square=True,
        cbar_kws={"shrink": 0.6},
    )
    plt.title("Metric Correlations", fontweight="bold", pad=15)
    # plt.xticks(rotation=90)
    plt.yticks(rotation=0, fontsize=9)

    # Ensure all y-ticks are shown (fix for missing y-ticks)
    ax.set_xticks(np.arange(len(short_names)) + 0.5)
    ax.set_xticklabels(short_names, rotation=90, fontsize=9)

    # 2. Normalized performance by category
    plt.subplot(2, 4, 2)
    category_scores = []
    model_names = []
    categories = []

    df["model_category"] = (
        df["type"] + " (" + df["optimized"].map({True: "Opt", False: "Base"}) + ")"
    )

    for model_idx in range(len(df)):
        model_name = df.iloc[model_idx]["model_category"]

        for cat_name, cat_info in METRIC_CATEGORIES.items():
            available_metrics = [m for m in cat_info["metrics"] if m in df.columns]
            if available_metrics:
                scores = [
                    get_performance_score(df, metric, model_idx)
                    for metric in available_metrics
                ]
                avg_score = np.mean(scores)
                category_scores.append(avg_score)
                model_names.append(model_name)
                categories.append(cat_info["title"])

    cat_df = pd.DataFrame(
        {"model": model_names, "category": categories, "score": category_scores}
    )

    # Truncate category names
    cat_df["category"] = cat_df["category"].apply(
        lambda x: x[:15] + "..." if len(x) > 18 else x
    )

    try:
        sns.boxplot(data=cat_df, x="category", y="score", palette="Set3")
    except:
        # Fallback for seaborn compatibility issues
        for i, cat in enumerate(cat_df["category"].unique()):
            cat_data = cat_df[cat_df["category"] == cat]["score"]
            plt.boxplot(cat_data, positions=[i], widths=0.6, patch_artist=True)
        plt.xticks(range(len(cat_df["category"].unique())), cat_df["category"].unique())

    plt.title("Performance by Category", fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Normalized Performance")
    plt.tick_params(axis="y")

    # 3. Type and optimization impact
    plt.subplot(2, 4, 3)
    df_analysis = df.copy()

    # Calculate overall performance score for each model
    overall_scores = []
    for model_idx in range(len(df)):
        scores = []
        for metric in metric_cols:
            score = get_performance_score(df, metric, model_idx)
            scores.append(score)
        overall_scores.append(np.mean(scores))

    df_analysis["overall_score"] = overall_scores

    sns.barplot(
        data=df_analysis, x="type", y="overall_score", hue="optimized", palette="Set2"
    )
    plt.legend(loc="upper right")

    plt.title("Performance by Type & Optimization", fontweight="bold", pad=15)
    plt.ylabel("Avg Normalized Score")
    plt.tick_params(axis="both")

    # 4. Model ranking by overall performance
    plt.subplot(2, 4, 4)
    df_ranked = df_analysis.sort_values("overall_score", ascending=True)

    colors = [
        "lightcoral" if not opt else "lightgreen" for opt in df_ranked["optimized"]
    ]

    bars = plt.barh(
        range(len(df_ranked)), df_ranked["overall_score"], color=colors, alpha=0.8
    )

    # Truncate model names for y-axis
    truncated_names = [
        name[:20] + "..." if len(name) > 23 else name
        for name in df_ranked["model_category"]
    ]
    plt.yticks(range(len(df_ranked)), truncated_names)
    plt.xlabel("Overall Performance Score")
    plt.title("Model Ranking", fontweight="bold", pad=15)
    plt.grid(axis="x", alpha=0.3)
    plt.tick_params(axis="x")

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="lightcoral", alpha=0.8, label="Not Optimized"),
        Patch(facecolor="lightgreen", alpha=0.8, label="Optimized"),
    ]
    plt.legend(handles=legend_elements, loc="lower right")

    # 5. Metric leaders by category
    plt.subplot(2, 4, 5)
    leader_counts = {}

    for cat_name, cat_info in METRIC_CATEGORIES.items():
        available_metrics = [m for m in cat_info["metrics"] if m in df.columns]
        for metric in available_metrics:
            best_model = get_best_model_for_metric(df, metric)
            leader_counts[best_model] = leader_counts.get(best_model, 0) + 1

    if leader_counts:
        leaders_df = pd.DataFrame(
            list(leader_counts.items()), columns=["model", "count"]
        )
        leaders_df = leaders_df.sort_values("count", ascending=True)

        # Truncate model names
        truncated_models = [
            name[:18] + "..." if len(name) > 21 else name
            for name in leaders_df["model"]
        ]

        plt.barh(
            range(len(leaders_df)),
            leaders_df["count"],
            color=plt.cm.tab10(np.arange(len(leaders_df))),
        )
        plt.yticks(range(len(leaders_df)), truncated_models)
        plt.xlabel("Number of Metrics Led")
        plt.title("Models Leading Most Metrics", fontweight="bold", pad=15)
        plt.grid(axis="x", alpha=0.3)
        plt.tick_params(axis="x")

    # 6. Category-wise performance heatmap
    plt.subplot(2, 4, 6)
    heatmap_data = []
    heatmap_models = []

    for model_idx in range(len(df)):
        model_name = df.iloc[model_idx]["model_category"]
        row_data = []
        for cat_name, cat_info in METRIC_CATEGORIES.items():
            available_metrics = [m for m in cat_info["metrics"] if m in df.columns]
            if available_metrics:
                scores = [
                    get_performance_score(df, metric, model_idx)
                    for metric in available_metrics
                ]
                avg_score = np.mean(scores)
            else:
                avg_score = 0
            row_data.append(avg_score)
        heatmap_data.append(row_data)
        heatmap_models.append(model_name)

    # Truncate names for heatmap
    truncated_models = [
        name[:15] + "..." if len(name) > 18 else name for name in heatmap_models
    ]
    truncated_categories = [
        cat_info["title"].split("(")[0].strip()[:12] + "..."
        if len(cat_info["title"].split("(")[0].strip()) > 15
        else cat_info["title"].split("(")[0].strip()
        for cat_info in METRIC_CATEGORIES.values()
    ]

    heatmap_df = pd.DataFrame(
        heatmap_data, index=truncated_models, columns=truncated_categories
    )

    sns.heatmap(
        heatmap_df,
        annot=True,
        cmap="RdYlGn",
        fmt=".2f",
        cbar_kws={"shrink": 0.6},
        annot_kws={"size": 8},
    )
    plt.title("Performance by Category & Model", fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    # 7. Optimization impact by category
    plt.subplot(2, 4, 7)
    opt_impact_data = []

    for cat_name, cat_info in METRIC_CATEGORIES.items():
        available_metrics = [m for m in cat_info["metrics"] if m in df.columns]
        if available_metrics:
            for optimized in [False, True]:
                subset = df[df["optimized"] == optimized]
                if len(subset) > 0:
                    scores = []
                    for model_idx in subset.index:
                        model_scores = [
                            get_performance_score(
                                df, metric, df.index.get_loc(model_idx)
                            )
                            for metric in available_metrics
                        ]
                        scores.extend(model_scores)

                    category_name = cat_info["title"].split("(")[0].strip()
                    truncated_cat = (
                        category_name[:12] + "..."
                        if len(category_name) > 15
                        else category_name
                    )

                    opt_impact_data.append(
                        {
                            "category": truncated_cat,
                            "optimized": "Optimized" if optimized else "Base",
                            "score": np.mean(scores),
                        }
                    )

    if opt_impact_data:
        opt_df = pd.DataFrame(opt_impact_data)
        sns.barplot(
            data=opt_df, x="category", y="score", hue="optimized", palette="Set1"
        )
        plt.legend(loc="upper right")

        plt.title("Optimization Impact by Category", fontweight="bold", pad=15)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Avg Performance Score")
        plt.tick_params(axis="y")

    # 8. Distribution of metric directions
    plt.subplot(2, 4, 8)
    direction_counts = {
        "Lower is Better": 0,
        "Higher is Better": 0,
        "Closer to Zero": 0,
    }

    for metric in metric_cols:
        direction = get_metric_direction(metric)
        if direction == "lower":
            direction_counts["Lower is Better"] += 1
        elif direction == "higher":
            direction_counts["Higher is Better"] += 1
        else:
            direction_counts["Closer to Zero"] += 1

    # Use shorter labels for pie chart
    short_labels = ["Lower Better", "Higher Better", "Closer to Zero"]
    plt.pie(
        direction_counts.values(),
        labels=short_labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=["lightcoral", "lightgreen", "lightblue"],
    )
    plt.title("Metric Direction Distribution", fontweight="bold", pad=15)

    # Save with tight layout and high DPI
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_4_metrics(df, metrics, save_path, figsize=(15, 12)):
    """
    Create a 2x2 grid comparing 4 selected metrics with bar plots.
    Properly handles different optimization directions.

    Args:
        df: DataFrame with model metrics
        metrics: List of 4 metric names to compare
        save_path: Where to save figure
        figsize: Figure size tuple
    """
    if len(metrics) != 4:
        raise ValueError("Please provide exactly 4 metrics")

    # Verify metrics exist in dataframe
    missing_metrics = [m for m in metrics if m not in df.columns]
    if missing_metrics:
        raise ValueError(f"Metrics not found in dataframe: {missing_metrics}")

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    # Create a combined category for better visualization
    df["category"] = (
        df["type"] + " (" + df["optimized"].map({True: "Opt", False: "Base"}) + ")"
    )

    # Color palette for categories
    categories = df["category"].unique()
    colors = sns.color_palette("Set2", len(categories))
    color_map = dict(zip(categories, colors))

    for i, metric in enumerate(metrics):
        ax = axes[i]
        direction = get_metric_direction(metric)

        # Sort by performance (not raw values)
        performance_scores = [
            get_performance_score(df, metric, idx) for idx in range(len(df))
        ]
        df_with_perf = df.copy()
        df_with_perf["performance"] = performance_scores
        df_sorted = df_with_perf.sort_values("performance", ascending=False)

        # Create bar plot with original metric values
        bars = ax.bar(
            range(len(df_sorted)),
            df_sorted[metric],
            color=[color_map[cat] for cat in df_sorted["category"]],
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )

        # Customize plot based on metric direction
        direction_indicator = {
            "lower": "↓ Lower is Better",
            "higher": "↑ Higher is Better",
            "zero": "→ Closer to Zero is Better",
        }

        ax.set_title(
            f"{metric}\n{direction_indicator[direction]}",
            fontsize=12,
            fontweight="bold",
            pad=20,
        )
        ax.set_xlabel("Models (Ranked by Performance)", fontsize=11)
        ax.set_ylabel(f"{metric} Value", fontsize=11)

        # Set x-axis labels
        ax.set_xticks(range(len(df_sorted)))
        ax.set_xticklabels(df_sorted["category"], rotation=45, ha="right", fontsize=9)

        # Add value labels on bars
        for j, (bar, val) in enumerate(zip(bars, df_sorted[metric])):
            height = bar.get_height()
            label_y = height + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                label_y,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

        # Add grid
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

        # Highlight best performer (leftmost bar, highest performance score)
        bars[0].set_edgecolor("gold")
        bars[0].set_linewidth(3)

    # Create shared legend
    legend_elements = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=color_map[cat],
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
            label=cat,
        )
        for cat in categories
    ]
    legend_elements.append(
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor="gold",
            alpha=0.8,
            edgecolor="gold",
            linewidth=3,
            label="Best Performer",
        )
    )

    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=len(legend_elements),
        fontsize=10,
    )

    plt.tight_layout()
    # plt.subplots_adjust(bottom=0.18)  # Increase bottom margin for legend
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def get_metric_suggestions(df, n=4):
    """
    Suggest interesting metric combinations based on correlation and variance,
    considering optimization directions.

    Args:
        df: DataFrame with model metrics
        n: Number of metrics to suggest
    """
    metric_cols = [
        col
        for col in df.columns
        if col not in ["model", "type", "optimized", "model_category"]
    ]
    log.info(metric_cols)

    # Calculate normalized performance variance to find most discriminative metrics
    normalized_data = {}
    for metric in metric_cols:
        normalized_values = []
        for idx in range(len(df)):
            score = get_performance_score(df, metric, idx)
            normalized_values.append(score)
        normalized_data[metric] = normalized_values

    norm_df = pd.DataFrame(normalized_data)
    variances = norm_df.var().sort_values(ascending=False)

    # Calculate correlation matrix on normalized data
    corr_matrix = norm_df.corr().abs()

    log.info("🔍 Metric Analysis (Direction-Aware):")
    log.info(f"Total metrics available: {len(metric_cols)}")

    # Show metrics by category
    for cat_name, cat_info in METRIC_CATEGORIES.items():
        available_metrics = [m for m in cat_info["metrics"] if m in df.columns]
        if available_metrics:
            log.info(f"\n{cat_info['title']}:")
            for metric in available_metrics:
                direction_symbol = {"lower": "↓", "higher": "↑", "zero": "→"}[
                    cat_info["better"]
                ]
                var_score = variances.get(metric, 0)
                log.info(f"  {direction_symbol} {metric}: variance={var_score:.3f}")

    log.info(f"\nTop {n} most discriminative metrics (normalized):")
    for i, (metric, var) in enumerate(variances.head(n).items(), 1):
        direction = get_metric_direction(metric)
        direction_symbol = {"lower": "↓", "higher": "↑", "zero": "→"}[direction]
        log.info(f"  {i}. {direction_symbol} {metric}: {var:.3f}")

    # Suggest diverse metric combinations (low correlation)
    log.info("\n💡 Suggested combinations for visualization:")
    high_var_metrics = variances.head(8).index.tolist()  # Top 8 most variable

    # Find combination with lowest average correlation
    best_combo = None
    best_score = float("inf")

    for combo in combinations(high_var_metrics, n):
        combo_corr = corr_matrix.loc[list(combo), list(combo)]
        # Average correlation excluding diagonal
        avg_corr = (combo_corr.sum().sum() - n) / (n * (n - 1))
        if avg_corr < best_score:
            best_score = avg_corr
            best_combo = combo

    # Add direction indicators to suggestions
    diverse_with_directions = []
    for metric in best_combo:
        direction = get_metric_direction(metric)
        direction_symbol = {"lower": "↓", "higher": "↑", "zero": "→"}[direction]
        diverse_with_directions.append(f"{direction_symbol} {metric}")

    log.info(f"  1. Diverse metrics (low correlation): {diverse_with_directions}")
    log.info(f"     Average correlation: {best_score:.3f}")

    # Suggest category-balanced combo
    category_balanced = []
    for cat_name, cat_info in METRIC_CATEGORIES.items():
        available_metrics = [m for m in cat_info["metrics"] if m in high_var_metrics]
        if available_metrics and len(category_balanced) < n:
            # Pick the most variable metric from this category
            best_from_category = max(available_metrics, key=lambda x: variances[x])
            category_balanced.append(best_from_category)

    # Fill remaining slots with most variable metrics
    remaining_slots = n - len(category_balanced)
    for metric in high_var_metrics:
        if metric not in category_balanced and remaining_slots > 0:
            category_balanced.append(metric)
            remaining_slots -= 1

    balanced_with_directions = []
    for metric in category_balanced:
        direction = get_metric_direction(metric)
        direction_symbol = {"lower": "↓", "higher": "↑", "zero": "→"}[direction]
        balanced_with_directions.append(f"{direction_symbol} {metric}")

    log.info(f"  2. Category-balanced metrics: {balanced_with_directions}")

    return list(best_combo), category_balanced


def plot_all_metrics(df, cols=4, figsize_per_plot=(4, 3), save_path=None):
    """
    Create a comprehensive grid showing ALL metrics with detailed bar plots.
    Similar to plot_4_metrics but for every metric in the dataset.

    Args:
        df: DataFrame with model metrics
        cols: Number of columns in the grid
        figsize_per_plot: Size of each individual plot (width, height)
        save_path: Optional path to save the figure
    """
    # Get metric columns (exclude model, type, optimized, category)
    metric_cols = [
        col
        for col in df.columns
        if col not in ["model", "type", "optimized", "category", "model_category"]
    ]
    log.info(metric_cols)

    df["model_category"] = (
        df["type"] + " (" + df["optimized"].map({True: "Opt", False: "Base"}) + ")"
    )

    if not metric_cols:
        log.warning("No metrics found in dataframe!")
        return

    n_metrics = len(metric_cols)
    rows = (n_metrics + cols - 1) // cols  # Ceiling division

    # Calculate figure size
    fig_width = cols * figsize_per_plot[0]
    fig_height = rows * figsize_per_plot[1]

    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))

    # Handle case where we have only one row or column
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]
    else:
        axes = axes

    # Flatten axes for easier indexing
    if rows > 1:
        axes_flat = [ax for row in axes for ax in row]
    else:
        axes_flat = axes if isinstance(axes, list) else [axes]

    # Create a combined category for better visualization
    df_work = df.copy()
    df_work["category"] = (
        df_work["type"]
        + " ("
        + df_work["optimized"].map({True: "Opt", False: "Base"})
        + ")"
    )

    # Color palette for categories
    categories = df_work["category"].unique()
    colors = sns.color_palette("Set2", len(categories))
    color_map = dict(zip(categories, colors))

    log.info("📊 Creating comprehensive metric visualization...")
    log.info(f"   • Total metrics: {n_metrics}")
    log.info(f"   • Grid size: {rows}x{cols}")
    log.info(f"   • Figure size: {fig_width:.1f}x{fig_height:.1f}")

    # Group metrics by category for organized display
    metric_order = []
    for cat_name, cat_info in METRIC_CATEGORIES.items():
        available_metrics = [m for m in cat_info["metrics"] if m in metric_cols]
        metric_order.extend(available_metrics)

    # Add any remaining metrics not in categories
    remaining_metrics = [m for m in metric_cols if m not in metric_order]
    metric_order.extend(remaining_metrics)

    metric_order = [
        m for m in metric_order if m != "category" and m != "model_category"
    ]

    for i, metric in enumerate(metric_order):
        if i >= len(axes_flat):
            break

        ax = axes_flat[i]
        direction = get_metric_direction(metric)

        # Sort by performance (not raw values)
        performance_scores = [
            get_performance_score(df_work, metric, idx) for idx in range(len(df_work))
        ]
        df_with_perf = df_work.copy()
        df_with_perf["performance"] = performance_scores
        df_sorted = df_with_perf.sort_values("performance", ascending=False)

        # Create bar plot with original metric values
        bars = ax.bar(
            range(len(df_sorted)),
            df_sorted[metric],
            color=[color_map[cat] for cat in df_sorted["category"]],
            alpha=0.8,
            edgecolor="black",
            linewidth=0.3,
        )

        # Customize plot based on metric direction
        direction_indicator = {"lower": "↓", "higher": "↑", "zero": "→"}

        # Shorter title for grid layout
        ax.set_title(
            f"{direction_indicator[direction]} {metric}",
            fontsize=10,
            fontweight="bold",
            pad=10,
        )

        # Smaller labels for grid layout
        ax.set_xlabel("Models", fontsize=8)
        ax.set_ylabel("Value", fontsize=8)

        # Set x-axis labels with smaller font
        ax.set_xticks(range(len(df_sorted)))
        ax.set_xticklabels(df_sorted["category"], rotation=45, ha="right", fontsize=7)

        # Add value labels on bars (smaller font)
        for j, (bar, val) in enumerate(zip(bars, df_sorted[metric])):
            height = bar.get_height()
            # Only show labels for top 3 performers to avoid clutter
            if j < 3:
                label_y = height + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    label_y,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    fontweight="bold",
                )

        # Highlight best performer
        bars[0].set_edgecolor("gold")
        bars[0].set_linewidth(2)

        # Adjust tick label sizes
        ax.tick_params(axis="both", which="major", labelsize=7)

        # Show which category this metric belongs to
        metric_category = None
        for cat_name, cat_info in METRIC_CATEGORIES.items():
            if metric in cat_info["metrics"]:
                metric_category = cat_name
                break

    # Hide empty subplots
    for i in range(n_metrics, len(axes_flat)):
        axes_flat[i].set_visible(False)

    # Create comprehensive legend
    legend_elements = []

    # Add category colors
    for cat in categories:
        legend_elements.append(
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor=color_map[cat],
                alpha=0.8,
                edgecolor="black",
                linewidth=0.5,
                label=cat,
            )
        )

    # Add special indicators
    legend_elements.extend(
        [
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor="gold",
                alpha=0.8,
                edgecolor="gold",
                linewidth=2,
                label="Best Performer",
            ),
        ]
    )

    # Place legend outside the plot area
    fig.legend(
        handles=legend_elements,
        loc="center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=len(legend_elements),
        fontsize=9,
    )

    # Add overall title
    fig.suptitle(
        f"All {n_metrics} Metrics Ranked by Performance",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12, top=0.92)  # Make room for legend and title

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        log.info(f"💾 Saved comprehensive metric plot to: {save_path}")

    plt.show()

    # Print summary statistics
    log.info("\n📈 Metric Summary:")
    log.info(
        f"   • Distance metrics (↓): {len([m for m in metric_cols if get_metric_direction(m) == 'lower'])}"
    )
    log.info(
        f"   • Quality metrics (↑): {len([m for m in metric_cols if get_metric_direction(m) == 'higher'])}"
    )
    log.info(
        f"   • Ratio metrics (→): {len([m for m in metric_cols if get_metric_direction(m) == 'zero'])}"
    )

    # Show top performer for each category
    log.info("\n🏆 Top Performers by Category:")
    for cat_name, cat_info in METRIC_CATEGORIES.items():
        available_metrics = [m for m in cat_info["metrics"] if m in df.columns]
        if available_metrics:
            category_leaders = {}
            for metric in available_metrics:
                best_model = get_best_model_for_metric(df, metric)
                category_leaders[best_model] = category_leaders.get(best_model, 0) + 1

            if category_leaders:
                top_model = max(category_leaders.items(), key=lambda x: x[1])
                log.info(
                    f"   • {cat_info['title']}: {top_model[0]} (leads {top_model[1]}/{len(available_metrics)} metrics)"
                )


def find_checkpoints():
    checkpoint_dir = Path(__file__).resolve().parent.parent / "model_checkpoints"
    matches = []
    for root, dirnames, filenames in os.walk(checkpoint_dir):
        for filename in fnmatch.filter(filenames, "*.ckpt"):
            matches.append(os.path.join(root, filename))
    return matches


def main():
    # Find appropriate values
    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = torch.cuda.device_count()
    elif torch.backends.mps.is_available():
        accelerator = "mps"
        devices = 1
    else:
        accelerator = "cpu"
        devices = "auto"

    strategy = "auto"
    trainer = pl.Trainer(accelerator=accelerator, devices=devices, strategy=strategy)
    device = trainer.strategy.root_device
    log.info(f"Using device {device}")

    # Set up real data comparison
    dataset_name = "Dmini/FFHQ-64x64"
    log.info(f"Loading dataset: {dataset_name}")
    real_data = load_huggingface_data(dataset_name, log, test=True, use_h5=True)
    log.info(real_data.shape)

    # Set up evaluator
    evaluator = StandaloneGenerativeModelEvaluator(logger=log, dataset_type="image")
    evaluator.cache_real_data(real_data, device)

    persistent_noise = torch.randn((16, 3, 64, 64), device=device)

    root_dir = Path(__file__).resolve().parent.parent
    eval_dir = root_dir / "final_evaluation_plots"
    eval_dir.mkdir(exist_ok=True)

    metrics_dict = {}

    checkpoints = find_checkpoints()
    for checkpoint in checkpoints:
        model_parts = checkpoint.split("/")[-4:]
        log.info(f"Creating samples for checkpoint {model_parts}")
        if model_parts[0] == "diffusion":
            model = DiffusionModel.load_from_checkpoint(checkpoint).to(device)
        else:
            model = FlowMatching.load_from_checkpoint(checkpoint).to(device)
        
        final_samples = model.sample_from_noise(persistent_noise, device)
        save_image_samples(
            final_samples, model_parts[0], "_".join(model_parts[1:]), eval_dir=eval_dir
        )
        log.info(f"Images saved for model {model_parts}. Now evaluating...")
        eval_metrics = evaluator.evaluate(final_samples)

        if "solution1" in model_parts[-2]:
            model_parts[-2] = model_parts[-2].replace("_solution1", "")
        elif "solution2" in model_parts[-2]:
            model_parts[-2] = f"{model_parts[-2]}_test"
        model_name = f"{model_parts[0]}_{model_parts[-2]}"
        metrics_dict[model_name] = eval_metrics

    df_full = pd.DataFrame.from_dict(metrics_dict, orient="index")
    df_full = df_full.reset_index().rename(columns={"index": "model"})
    df_full["type"] = df_full["model"].apply(lambda x: "fm" if "flow" in x else "diff")
    df_full["optimized"] = df_full["model"].apply(
        lambda x: True if "optim" in x else False
    )
    df_full["test"] = df_full["model"].apply(
        lambda x: True if any(sub in x for sub in ["test", "short"]) else False
    )
    print(df_full.head())
    df = df_full[df_full["test"] == False]
    df = df.drop(columns=["test"])
    print(df.head())

    plot_overview(df, eval_dir / "sample_metrics_overview.pdf")

    diverse_metrics, balanced_metrics = get_metric_suggestions(df)
    log.info(f"Diverse metrics: {diverse_metrics}")
    log.info(f"Balanced metrics: {balanced_metrics}")

    selected_metrics = ["fid", "coverage", "precision", "distance_entropy"]
    plot_4_metrics(df, selected_metrics, eval_dir / "selected_metrics_comparison.pdf")

    plot_all_metrics(df, cols=5, save_path=eval_dir / "complete_metrics_comparison.pdf")

    df.to_csv(eval_dir / "final_results.csv")


if __name__ == "__main__":
    main()
