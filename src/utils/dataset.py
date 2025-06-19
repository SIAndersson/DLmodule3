import torch
from sklearn.datasets import make_moons
from omegaconf import DictConfig
import logging
from datasets import load_dataset
from torchvision import transforms
import random
from dotenv import load_dotenv
import os

load_dotenv()

HF_DATASETS_CACHE = os.getenv("HF_DATASETS_CACHE")


def load_huggingface_data(
    dataset_name: str, logger, test: bool = False
) -> torch.Tensor:
    """
    Load a dataset from Huggingface and transform it into a PyTorch Dataset
    that returns image tensors.

    NOTE: The dataset should have an "image" column with PIL images or paths to images.

    Returns:
        data_tensor (Tensor): A PyTorch Tensor containing image data.
    """

    # Load dataset from Huggingface
    dataset = load_dataset(
        dataset_name, split="train", cache_dir=HF_DATASETS_CACHE
    )  # or "test"

    logger.info(f"Dataset loaded from {dataset_name}.")

    # Truncate dataset if test is True
    if test:
        random_indices = random.sample(range(len(dataset)), 4000)
        dataset = dataset.select(random_indices)
        logger.info("Dataset truncated to length 4000.")

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
        num_proc=4,
    )

    logger.info("Dataset transformed.")

    # Get all pixel values and stack them
    all_pixel_values = dataset["pixel_values"]

    # Concatenate all batches into a single tensor
    all_tensors = torch.tensor(all_pixel_values)  # Final shape (N, C, H, W)
    logger.info(f"Min: {all_tensors.min()}, Max: {all_tensors.max()}")
    logger.debug(f"Final shape: {all_tensors.shape}")

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
        return load_huggingface_data(dataset_name, log, cfg.main.test)
    else:
        raise ValueError(f"Unknown dataset: {cfg.main.dataset}")
