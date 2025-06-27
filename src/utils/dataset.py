import logging
import os
import random
from pathlib import Path

import h5py
import numpy as np
import math
from tqdm import tqdm
import pytorch_lightning as pl
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from omegaconf import DictConfig
from sklearn.datasets import make_moons
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

load_dotenv()

HF_DATASETS_CACHE = os.getenv("HF_DATASETS_CACHE")


def load_huggingface_data(
    dataset_name: str, logger, test: bool = False, use_h5: bool = False
) -> torch.Tensor:
    """
    Load a dataset from Huggingface and transform it into a PyTorch Dataset
    that returns image tensors.

    NOTE: The dataset should have an "image" column with PIL images or paths to images.

    Returns:
        data_tensor (Tensor): A PyTorch Tensor containing image data.
    """

    # Obtain dataset file name
    dataset_base = dataset_name.split("/")[-1]
    filename = dataset_base
    if test:
        filename += "_test"
    save_path = Path(__file__).parent.parent.parent / "data" / f"{filename}.{'h5' if use_h5 else 'pt'}"

    # Check if dataset file exists
    if save_path.exists():
        logger.info(f"Loading dataset tensor from {save_path}")
        if use_h5:
            with h5py.File(save_path, 'r') as f:
                data = f['images'][:]
                return torch.from_numpy(data)
        else:
            all_tensors = torch.load(save_path)
            return all_tensors

    # Load dataset from Huggingface
    dataset = load_dataset(
        dataset_name, split="train", cache_dir=HF_DATASETS_CACHE
    )  # or "test"

    logger.info(f"Dataset loaded from {dataset_name}.")

    # Truncate dataset if test is True
    if test:
        random_indices = random.sample(range(len(dataset)), 6000)
        dataset = dataset.select(random_indices)
        logger.info("Dataset truncated to length 6000.")

    # Set transforms (currently just converts to tensor)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # (C, H, W)
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
            ),  # Normalize to [-1, 1]
        ]
    )

    # Transform in batches for speed
    def transform_batch(batch):
        batch["pixel_values"] = [
            transform(img.convert("RGB")) for img in batch["image"]
        ]
        return batch

    logger.info("Transforming dataset...")

    dataset = dataset.map(
        transform_batch,
        batched=True,
        remove_columns=["image"],
        batch_size=1000,
        num_proc=10,
    )

    logger.info("Dataset transformed.")
    dataset.set_format(type="torch", columns=["pixel_values"])

    logger.info("Dataset transformed.")

    N = len(dataset)

    sample_batch = dataset[0:1]["pixel_values"]
    _, C, H, W = sample_batch.shape
    dtype  = sample_batch.dtype
    device = sample_batch.device

    all_tensors = torch.empty((N, C, H, W), dtype=dtype, device=device)

    chunk_size = 1000
    num_chunks = math.ceil(N / chunk_size)

    for i in tqdm(range(num_chunks), desc="Loading in dataset..."):
        start = i * chunk_size
        end   = min(start + chunk_size, N)
        chunk_tensor = dataset[start:end]["pixel_values"]
        all_tensors[start:end] = chunk_tensor

    # Save tensor to data directory
    logger.info(f"Saving dataset to {save_path}...")
    if use_h5:
        # Convert to numpy for saving
        numpy_data = all_tensors.cpu().numpy()
        with h5py.File(save_path, 'w') as f:
            f.create_dataset('images', data=numpy_data, compression='gzip', compression_opts=1)
    else:
        torch.save(all_tensors, save_path)
    logger.info(f"Dataset saved successfully!")

    logger.info(f"Final tensor - Shape: {all_tensors.shape}, Min: {all_tensors.min():.3f}, Max: {all_tensors.max():.3f}")

    return all_tensors


def create_2d_dataset(n_samples=10000):
    """
    Create a 2D mixture of Gaussians dataset for visualization
    This creates a simple but interesting distribution to model
    """
    # Create mixture of 4 Gaussians in 2D
    centers = torch.tensor([[-2, -2], [-2, 2], [2, -2], [2, 2]], dtype=torch.float32)
    n_per_cluster = n_samples // 4

    data = []
    for center in centers:
        cluster_data = torch.randn(n_per_cluster, 2) * 0.5 + center
        data.append(cluster_data)

    return torch.cat(data, dim=0)


def create_two_moons_data(n_samples):
    X, _ = make_moons(n_samples=n_samples, noise=0.1, random_state=42)
    return torch.FloatTensor(X)


def create_dataset(cfg: DictConfig, log: logging.Logger):
    """
    Create dataset based on configuration.
    """
    if cfg.main.dataset.lower() == "two_moons":
        log.info("Creating Two Moons dataset...")
        return create_two_moons_data(cfg.main.num_samples)
    elif cfg.main.dataset.lower() == "2d_gaussians":
        log.info("Creating 2D Gaussian mixture dataset...")
        return create_2d_dataset(cfg.main.num_samples)
    elif cfg.main.dataset.lower() == "ffhq":
        dataset_name = "bitmind/ffhq-256"
        log.info(f"Loading dataset: {dataset_name}")
        return load_huggingface_data(dataset_name, log, cfg.main.test, cfg.main.get("use_h5", False))
    else:
        raise ValueError(f"Unknown dataset: {cfg.main.dataset}")


class GenerativeDataModule(pl.LightningDataModule):
    def __init__(self, cfg, logger):
        super().__init__()
        self.cfg = cfg
        self.logger = logger
        self.data_tensor = None

    # If dataset requires downloading, do this on ONE core so that it is ready
    def prepare_data(self):
        if (
            self.cfg.main.dataset.lower() != "two_moons"
            and self.cfg.main.dataset.lower() != "2d_gaussians"
        ):
            dataset_name = "bitmind/ffhq-256"
            self.logger.info(f"Preparing data for {dataset_name}.")
            dataset_base = dataset_name.split("/")[-1]
            filename = dataset_base
            if self.cfg.main.test:
                filename += "_test"
            save_path = Path(__file__).parent.parent.parent / "data" / f"{filename}.pt"

            # Check if dataset file exists
            if not save_path.exists():
                _ = create_dataset(self.cfg, self.logger)

    # Setup only runs once per GPU
    def setup(self, stage=None):
        if self.data_tensor is None:
            self.data_tensor = create_dataset(self.cfg, self.logger)

        # Create train/val split
        val_size = getattr(self.cfg.main, "val_size", 0.1)  # Default to 0.1 if not set
        dataset = TensorDataset(self.data_tensor)
        n_total = len(dataset)
        n_val = int(n_total * val_size)
        n_train = n_total - n_val
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(self.cfg.main.seed),
        )

    # Lightning automatically sorts distributed sampler etc
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=32
            if self.cfg.main.gradient_accumulation
            else self.cfg.main.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=32
            if self.cfg.main.gradient_accumulation
            else self.cfg.main.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

    def get_original_data(self):
        """Access the original data for comparison"""
        return self.data_tensor
