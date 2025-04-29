from typing import Dict
import torch
from torch.utils.data import Dataset

class RLDataset(Dataset):
    """
    A class to manage a buffer of data for reinforcement learning.
    """
    def __init__(
        self,
        *,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        super(RLDataset, self).__init__()
        self.device = device
        self.dtype = dtype
        self.data = {}
        
    def add(self, **rhs) -> "RLDataset":
        """
        Add data to the buffer, concatenating tensors along the first dimension.
        Args:
            rhs (Dict[str, List[torch.Tensor]]): A dictionary where keys are strings and values
                Non tensor values or lists of tensors are converted to tensors.
        """
        for key, value in rhs.items():
            if not isinstance(value, torch.Tensor):
                # Single non-tensor value
                value = torch.tensor(value)
            # Tensor value, ensure it is at least 2D
            value = torch.atleast_2d(value)
            # Flatten all the dimensions except the last one:
            value = value.flatten(start_dim=0, end_dim=-2)
            if not key in self.data:
                # If the key does not exist in the buffer, create it
                self.data[key] = value
            else:
                # If the key exists, concatenate the new value to the existing tensor
                self.data[key] = torch.cat([self.data[key], value], dim=0)
            # Ensure all tensors are on the same device and dtype
            # ! Also detaches from the computation graph
            self.data[key] = self.data[key].to(
                device=self.device, dtype=self.dtype
            ).detach()

        return self

    def __getattr__(self, key):
        """
        Get an item from the buffer.
        Args:
            key (str): The key to get from the buffer.
        Returns:
            torch.Tensor: The tensor associated with the key.
        """
        if key in self.data:
            return self.data[key]
        else:
            raise AttributeError(f"Key '{key}' not found in buffer.")

    def __len__(self):
        """
        Get the length of the buffer.
        """
        if not self.data: 
            return 0
        else:
            key = list(self.data.keys())[0]
            return len(self.data[key])

    def __getitem__(self, indices) -> Dict[str, torch.Tensor]:
        """
        Get a mini-batch of data from the buffer.
        Args:
            indices (torch.Tensor): Indices to sample from the buffer.
        Returns:
            DatasetPPO: A new DatasetPPO object containing the sampled data.
        """
        sampled_data = {}
        for key, value in self.data.items():
            sampled_data[key] = value[indices]
        return sampled_data
