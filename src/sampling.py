import logging
import os
import fnmatch

from typing import Dict, Tuple

import sys
import colorlog
from pathlib import Path
import yaml
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import MLFlowLogger
from torch.optim import AdamW

from utils.callbacks import (
    EvaluationMixin,
    MetricTracker,
    create_early_stopping_callback,
    create_evaluation_config,
    create_model_checkpoint_callback,
)
from utils.evaluator import StandaloneGenerativeModelEvaluator
from utils.dataset import GenerativeDataModule, load_huggingface_data
from utils.models import CNN, MLP
from utils.seeding import set_seed
from utils.visualisation import (
    create_metrics_summary_table,
    plot_evaluation_metrics,
    plot_loss_function,
    save_2d_samples,
    save_image_samples,
    visualize_diffusion_process,
    plot_final_metrics,
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

if torch.cuda.is_available():
    # Get properties of the first available GPU
    device_props = torch.cuda.get_device_properties(0)
    if device_props.major >= 7:
        torch.set_float32_matmul_precision("high")
        log.info("Tensor cores enabled globally")
        
        
def create_comprehensive_comparison(metrics_dict: Dict[str, Dict[str, float]],
                                   normalize_metrics: bool = True,
                                   figsize: Tuple[int, int] = (20, 15)) -> plt.Figure:
    """
    Create a comprehensive multi-panel visualization optimized for many metrics.
    
    Args:
        metrics_dict: Dictionary with model names as keys and metric dictionaries as values
        normalize_metrics: Whether to normalize metrics to 0-1 scale for better comparison
        figsize: Figure size tuple
    """
    fig = plt.figure(figsize=figsize)
    
    # Extract data
    model_names = list(metrics_dict.keys())
    all_metrics = sorted(set().union(*[m.keys() for m in metrics_dict.values()]))
    
    # Prepare data matrix
    data_matrix = []
    for model in model_names:
        row = [metrics_dict[model].get(metric, np.nan) for metric in all_metrics]
        data_matrix.append(row)
    data_matrix = np.array(data_matrix)
    
    # Normalize if requested
    if normalize_metrics:
        # Handle NaN values and normalize each metric to 0-1 scale
        normalized_data = []
        for j in range(len(all_metrics)):
            col = data_matrix[:, j]
            col_clean = col[~np.isnan(col)]
            if len(col_clean) > 0:
                col_min, col_max = col_clean.min(), col_clean.max()
                if col_max > col_min:
                    col_norm = (col - col_min) / (col_max - col_min)
                else:
                    col_norm = np.zeros_like(col)
            else:
                col_norm = col
            normalized_data.append(col_norm)
        data_matrix = np.array(normalized_data).T
    
    # Create subplots
    gs = fig.add_gridspec(3, 4, height_ratios=[2, 1, 1], width_ratios=[3, 1, 1, 1], 
                         hspace=0.3, wspace=0.3)
    
    # 1. Main heatmap (top-left, large)
    ax1 = fig.add_subplot(gs[0, :3])
    im = ax1.imshow(data_matrix, cmap='RdYlBu_r', aspect='auto')
    ax1.set_xticks(range(len(all_metrics)))
    ax1.set_xticklabels(all_metrics, rotation=45, ha='right')
    ax1.set_yticks(range(len(model_names)))
    ax1.set_yticklabels(model_names)
    ax1.set_title('Complete Metrics Heatmap', fontsize=14, fontweight='bold')
    
    # Add values to heatmap
    for i in range(len(model_names)):
        for j in range(len(all_metrics)):
            if not np.isnan(data_matrix[i, j]):
                color = 'white' if data_matrix[i, j] > 0.5 else 'black'
                ax1.text(j, i, f'{data_matrix[i, j]:.2f}', 
                        ha='center', va='center', color=color, fontsize=8)
    
    # 2. Overall performance ranking (top-right)
    ax2 = fig.add_subplot(gs[0, 3])
    # Calculate average performance (handling NaN)
    avg_scores = np.nanmean(data_matrix, axis=1)
    colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
    bars = ax2.barh(range(len(model_names)), avg_scores, color=colors)
    ax2.set_yticks(range(len(model_names)))
    ax2.set_yticklabels(model_names)
    ax2.set_xlabel('Avg Score')
    ax2.set_title('Overall Ranking', fontweight='bold')
    for i, (bar, score) in enumerate(zip(bars, avg_scores)):
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{score:.3f}', va='center', fontsize=9)
    
    # 3. Metric variance analysis (bottom-left)
    ax3 = fig.add_subplot(gs[1, :2])
    metric_vars = np.nanvar(data_matrix, axis=0)
    ax3.bar(range(len(all_metrics)), metric_vars, color='skyblue', alpha=0.7)
    ax3.set_xticks(range(len(all_metrics)))
    ax3.set_xticklabels(all_metrics, rotation=45, ha='right')
    ax3.set_ylabel('Variance')
    ax3.set_title('Metric Discriminative Power', fontweight='bold')
    
    # 4. Model similarity matrix (bottom-middle)
    ax4 = fig.add_subplot(gs[1, 2])
    # Calculate pairwise correlation between models
    model_corr = np.corrcoef(data_matrix)
    im4 = ax4.imshow(model_corr, cmap='coolwarm', vmin=-1, vmax=1)
    ax4.set_xticks(range(len(model_names)))
    ax4.set_xticklabels([name[:8] + '...' if len(name) > 8 else name for name in model_names], 
                       rotation=45, ha='right', fontsize=8)
    ax4.set_yticks(range(len(model_names)))
    ax4.set_yticklabels([name[:8] + '...' if len(name) > 8 else name for name in model_names], 
                       fontsize=8)
    ax4.set_title('Model Similarity', fontweight='bold', fontsize=10)
    
    # 5. Top/Bottom performers per metric (bottom-right)
    ax5 = fig.add_subplot(gs[1, 3])
    ax5.axis('off')
    ax5.text(0.5, 0.9, 'Best Performers', ha='center', fontweight='bold', 
             transform=ax5.transAxes, fontsize=12)
    
    # Find best performer for each metric
    y_pos = 0.8
    for i, metric in enumerate(all_metrics[:8]):  # Show top 8 metrics
        if not np.isnan(data_matrix[:, i]).all():
            best_idx = np.nanargmax(data_matrix[:, i])
            best_model = model_names[best_idx]
            best_score = data_matrix[best_idx, i]
            ax5.text(0.05, y_pos, f'{metric[:12]}:', fontsize=8, 
                    transform=ax5.transAxes, fontweight='bold')
            ax5.text(0.95, y_pos, f'{best_model[:8]} ({best_score:.2f})', 
                    fontsize=8, transform=ax5.transAxes, ha='right')
            y_pos -= 0.09
    
    # 6. Worst performers (bottom part of same subplot)
    ax5.text(0.5, 0.35, 'Worst Performers', ha='center', fontweight='bold', 
             transform=ax5.transAxes, fontsize=12, color='red')
    
    y_pos = 0.25
    for i, metric in enumerate(all_metrics[:8]):
        if not np.isnan(data_matrix[:, i]).all():
            worst_idx = np.nanargmin(data_matrix[:, i])
            worst_model = model_names[worst_idx]
            worst_score = data_matrix[worst_idx, i]
            ax5.text(0.05, y_pos, f'{metric[:12]}:', fontsize=8, 
                    transform=ax5.transAxes, fontweight='bold')
            ax5.text(0.95, y_pos, f'{worst_model[:8]} ({worst_score:.2f})', 
                    fontsize=8, transform=ax5.transAxes, ha='right', color='red')
            y_pos -= 0.09
    
    # Add colorbars
    cbar1 = plt.colorbar(im, ax=ax1, shrink=0.6)
    cbar1.set_label('Normalized Score' if normalize_metrics else 'Raw Score')
    
    cbar4 = plt.colorbar(im4, ax=ax4, shrink=0.6)
    cbar4.set_label('Correlation')
    
    plt.suptitle('Generative Model Evaluation: Comprehensive Analysis', 
                fontsize=16, fontweight='bold')
    
    return fig

def find_checkpoints():
    checkpoint_dir = Path(__file__).resolve().parent.parent / 'model_checkpoints'
    matches = []
    for root, dirnames, filenames in os.walk(checkpoint_dir):
        for filename in fnmatch.filter(filenames, '*.ckpt'):
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
    
    # Set up evaluator
    evaluator = StandaloneGenerativeModelEvaluator(logger=log, dataset_type="image")
    evaluator.cache_real_data(real_data, device)

    persistent_noise = torch.randn((16, 3, 64, 64), device=device)
    
    metrics_dict = {}

    checkpoints = find_checkpoints()
    for checkpoint in checkpoints:
        model_parts = checkpoint.split('/')[-4:]
        if model_parts[0] == 'diffusion':
            model = DiffusionModel.load_from_checkpoint(checkpoint).to(device)
        else:
            model = FlowMatching.load_from_checkpoint(checkpoint).to(device)
        log.info(f'Creating samples for checkpoint {model_parts}')
        final_samples = model.sample_from_noise(persistent_noise, device)
        save_image_samples(final_samples, model_parts[0], '_'.join(model_parts[1:]))
        log.info(f"Images saved for model {model_parts}. Now evaluating...")
        eval_metrics = evaluator.evaluate(final_samples)
        metrics_dict['_'.join(model_parts[:-1])] = eval_metrics
        
    fig_comprehensive = create_comprehensive_comparison(metrics_dict, 
                                                      normalize_metrics=True)
    # Save image
    root_dir = Path(__file__).resolve().parent.parent
    eval_dir = root_dir / "evaluation_plots"
    eval_dir.mkdir(exist_ok=True)
    plt.savefig(eval_dir / "final_sample_metrics_comparison.pdf")
    plt.close()

if __name__ == "__main__":
    main()