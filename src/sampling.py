import logging
import os
import fnmatch

from typing import Dict

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
from utils.dataset import GenerativeDataModule
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

if torch.cuda.is_available():
    # Get properties of the first available GPU
    device_props = torch.cuda.get_device_properties(0)
    if device_props.major >= 7:
        torch.set_float32_matmul_precision("high")
        log.info("Tensor cores enabled globally")

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

    persistent_noise = torch.randn((16, 3, 64, 64), device=device)

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

if __name__ == "__main__":
    main()