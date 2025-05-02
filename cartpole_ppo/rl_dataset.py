from typing import Dict
import torch
from torch.utils.data import Dataset
from .convert_to_tensor import convert_to_tensor

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
            # Convert the value to a tensor if it is not already
            value = convert_to_tensor(value, device=self.device, dtype=self.dtype)
            if not key in self.data:
                # If the key does not exist in the buffer, create it
                self.data[key] = value
            else:
                # If the key exists, concatenate the new value to the existing tensor
                self.data[key] = torch.cat([self.data[key], value], dim=0)
        # Move the buffer to the specified device and dtype
        self.to(device=self.device, dtype=self.dtype)
        return self

    def to(self, device: torch.device, dtype: torch.dtype) -> "RLDataset":
        """
        Move the buffer to a different device and dtype.
        Args:
            device (torch.device): The device to move the buffer to.
            dtype (torch.dtype): The dtype to convert the buffer to.
        Returns:
            RLDataset: The modified dataset.
        """
        for key in self.data:
            self.data[key] = self.data[key].to(device=device, dtype=dtype)
        return self

    def detach(self) -> "RLDataset":
        """
        Detach the tensors in the buffer from the computation graph.
        Returns:
            RLDataset: The modified dataset.
        """
        for key in self.data:
            self.data[key] = self.data[key].detach()
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

    def __setattr__(self, key, value):
        """
        Set an item in the buffer.
        Args:
            key (str): The key to set in the buffer.
            value (torch.Tensor): The tensor to set.
        """
        self_keys = ["data", "device", "dtype"]
        if key in self_keys:
            super().__setattr__(key, value)
        else:
            value = convert_to_tensor(
                value, device=self.device, dtype=self.dtype
            )
            self.data[key] = value

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
