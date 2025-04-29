from typing import Optional, List
import torch

class PolicyBuffer:
    """
    Buffer to store policy data for PPO.
    """

    # Which keys are only used during rollout and should be removed
    # Rewards not used in validation of the policy, but used during testing
    # of models performance
    rollout_only_keys = {"values", "dones"}

    def __init__(self, **kwargs) -> None:
        self.states: List[torch.Tensor] = []
        self.actions_normal: List[torch.Tensor] = []
        self.actions_squashed: List[torch.Tensor] = []
        self.log_probs_squashed: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.returns: List[torch.Tensor] = []
        self.advantages: List[torch.Tensor] = []
        self.dones: List[torch.Tensor] = []
        for key, value in kwargs.items():
            if key in vars(self):
                setattr(self, key, value)

    def _stack_if_list(self, value):
        if isinstance(value, list) and value:
            assert all(isinstance(v, torch.Tensor) for v in value), "All elements must be tensors."
            return torch.stack(value)
        return value

    def to_tensor_(
        self,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Convert buffer data to tensors in-place.
        """
        for key in vars(self):
            val = getattr(self, key)
            val = self._stack_if_list(val)
            if isinstance(val, torch.Tensor):
                val = val.to(device=device, dtype=dtype).detach()
            setattr(self, key, val)
        return self

    def sanitize_(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Convert to tensors and remove rollout-only data.
        """
        self.to_tensor_(device=device, dtype=dtype)
        for key in self.rollout_only_keys:
            if hasattr(self, key):
                delattr(self, key)
        return self

    def add_buffer_(
        self,
        rhs: "PolicyBuffer",
        concat_dim: int = 0,
    ):
        """
        Concatenate another buffer to this buffer along the given dimension.
        """
        # if current buffer is empty, just return the rhs buffer
        if not len(self.states):
            for key in vars(rhs):
                val = getattr(rhs, key)
                setattr(self, key, val)
            return self.sanitize_()

        self.sanitize_()
        rhs.sanitize_()

        for key in vars(self):
            rhs_val = getattr(rhs, key, None)
            if rhs_val is None:
                raise ValueError(f"Key '{key}' not found in rhs buffer.")

            lhs_val = getattr(self, key)
            if isinstance(lhs_val, torch.Tensor) and isinstance(rhs_val, torch.Tensor):
                setattr(self, key, torch.cat([lhs_val, rhs_val], dim=concat_dim))
            else:
                raise TypeError(f"Cannot concatenate non-tensor field '{key}'.")

        return self


class BufferDataset:
    """
    Dataset class for the policy buffer.
    """

    def __init__(self, buffer: PolicyBuffer, permute: bool = True):
        self.buffer = buffer
        self.permute = permute
        if permute:
            self.permutation = torch.randperm(len(buffer.states))

    def __len__(self):
        return len(self.buffer.states)

    def __getitem__(self, idx):
        if self.permute:
            idx = self.permutation[idx]
        return PolicyBuffer(**{
            key: getattr(self.buffer, key).flatten(0, -2)[idx]
            for key in vars(self.buffer)
        })

    def get_minibatch(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Get a mini-batch of data from the buffer.
        """
        flattened_len = self.buffer.states.flatten(0, -2).shape[0]
        indices = torch.randint(0, flattened_len, (batch_size,))
        batch = self.__getitem__(indices)
        return batch.sanitize_(device=device, dtype=dtype)
