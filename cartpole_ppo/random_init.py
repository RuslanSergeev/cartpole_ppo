import numpy as np
import torch
import random

def set_seed(seed: int = 42):
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): The seed value to set.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
