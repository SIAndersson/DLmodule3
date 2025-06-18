import torch
from datasets import load_dataset
from torchvision import transforms


def load_huggingface_data(dataset_name: str) -> torch.Tensor:
    """
    Load a dataset from Huggingface and transform it into a PyTorch Dataset
    that returns image tensors.

    NOTE: The dataset should have an "image" column with PIL images or paths to images.

    Returns:
        tensor_dataset (Dataset): A PyTorch Dataset containing image tensors.
    """

    # Load dataset from Huggingface
    dataset = load_dataset(dataset_name, split="train")  # or "test"

    # Set transforms (currently just converts to tensor)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # (C, H, W)
        ]
    )

    # Transform in batches for speed
    def transform_batch(batch):
        batch["pixel_values"] = [
            transform(img.convert("RGB")) for img in batch["image"]
        ]
        return batch

    dataset = dataset.map(
        transform_batch,
        batched=True,
        remove_columns=["image"],
        batch_size=1000,
        num_proc=4,
    )
    # Each item in `dataset` is a dict: {"pixel_values": Tensor}

    all_tensors = torch.stack(dataset["pixel_values"])

    return all_tensors
