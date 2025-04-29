import torch
from torch import Dataset


class DatasetPPO(Dataset):
    """
    A class to manage a buffer of data for reinforcement learning.
    """
    def __init__(
        self,
        *,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        super(DatasetPPO, self).__init__()
        self.device = device
        self.dtype = dtype
        self.buffer = {}
        
    def add(self, **rhs) -> "DatasetPPO":
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
            if not key in self.buffer:
                # If the key does not exist in the buffer, create it
                self.buffer[key] = value
            else:
                # If the key exists, concatenate the new value to the existing tensor
                self.buffer[key] = torch.cat([self.buffer[key], value], dim=0)
            # Ensure all tensors are on the same device and dtype
            # ! Also detaches from the computation graph
            self.buffer[key] = self.buffer[key].to(device=self.device, dtype=self.dtype).detach()

        return self

    def __len__(self):
        """
        Get the length of the buffer.
        """
        if not self.buffer: 
            return 0
        else:
            key = list(self.buffer.keys())[0]
            return len(self.buffer[key])

    def __getitem__(self, indices) -> "DatasetPPO":
        """
        Get a mini-batch of data from the buffer.
        Args:
            indices (torch.Tensor): Indices to sample from the buffer.
        Returns:
            DatasetPPO: A new DatasetPPO object containing the sampled data.
        """
        sampled_data = {}
        for key, value in self.buffer.items():
            sampled_data[key] = value[indices]
        return DatasetPPO(device=self.device, dtype=self.dtype).add(**sampled_data)

    def get_minibatch(
        self,
        batch_size: int,
    ) -> "DatasetPPO":
        """
        Get a mini-batch of data from the buffer.
        Args:
            batch_size (int): The size of the mini-batch to retrieve.
        Returns:
            DatasetPPO: A new DatasetPPO object containing the mini-batch.
            If batch_size is less than or equal to 0, return the original buffer.
        """
        if batch_size <= 0:
            return self
        if batch_size > len(self):
            raise ValueError("Batch size exceeds buffer length.")
        indices = torch.randint(0, len(self), (batch_size,))
        return self.__getitem__(indices)
