import os
import random

import numpy as np
import torch
from lightning.pytorch import seed_everything


def set_seed(seed: int):
    """
    Seed everything for exact reproducibility across PyTorch, NumPy, random,
    and CUDA (if available). Use before any model / DataLoader / sampler instantiation.
    """
    # 1) Python
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # 2) NumPy
    np.random.seed(seed)
    # 3) PyTorch CPU
    torch.manual_seed(seed)
    # 4) PyTorch CUDA (if you ever use GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 5) Torch backend flags for determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    seed_everything(seed, workers=True)
