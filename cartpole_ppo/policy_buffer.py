from typing import Optional, Dict
import torch


class PolicyBuffer:
    """
    Buffer to store policy data for PPO.
    """

    # Which keys are only used during rollout and should be removed
    # Rewards not used in validation of the policy, but used during testing
    # of models performance
    rollout_only_keys = ("values", "dones")
    buffered_keys = (
        "states", 
        "actions_normal",
        "actions_squashed",
        "log_probs_squashed",
        "rewards",
        "returns",
        "advantages",
    )
    
    def __init__(self, device, dtype) -> None:
        for k in self.buffered_keys + self.rollout_only_keys:
            setattr(self, k, [])
        self.sanitized = False
        self.device = device
        self.dtype = dtype

    def append(self, data: Dict[str, torch.Tensor]) -> "PolicyBuffer":
        """
        Append data to the buffer.
        """
        for key, value_rhs in data.items():
            if hasattr(self, key):
                value_lhs = getattr(self, key)
                if not isinstance(value_rhs, torch.Tensor):
                    value_rhs = torch.tensor(value_rhs, device=self.device, dtype=self.dtype)
                value_rhs = value_rhs\
                    .to(device=self.device, dtype=self.dtype)\
                    .atleast_1d()
                if not self.sanitized:
                    value_lhs.append(value_rhs)
                else:
                    value_lhs = torch.cat([value_lhs, value_rhs], dim=0)\
                        .to(device=self.device, dtype=self.dtype)
            else:
                raise ValueError(f"Key '{key}' not found in PolicyBuffer.")
        return self

    def __len__(self) -> int:
        return len(getattr(self, self.buffered_keys[0]))

    def to_tensor_(
        self,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "PolicyBuffer":
        """
        Convert buffer data to tensors in-place.
        """
        for key in vars(self):
            value = getattr(self, key)
            if isinstance(value, list):
                value = torch.stack(value, dim=0)
            value = value.to(device=device, dtype=dtype)
            setattr(self, key, value)
        self.sanitized = True
        return self

    def append_(
        self,
        rhs: "PolicyBuffer",
        concat_dim: int = 0,
    ):
        """
        Concatenate another buffer to this buffer along the given dimension.
        """
        lhs_sanitized_or_empty = self.sanitized or not len(self)
        if not lhs_sanitized_or_empty:
            raise RuntimeError("Buffer must be sanitized or empty before concatenation.")
        if not rhs.sanitized:
            raise RuntimeError("Buffer must be sanitized before concatenation.")
        # Move the rhs to the same device and dtype as self
        rhs.to_tensor_(device=self.device, dtype=self.dtype)
        # if current buffer is empty, just copy the values
        if not len(self):
            for key in vars(rhs):
                val = getattr(rhs, key)
                if isinstance(val, torch.Tensor):
                    setattr(self, key, val.to(device=self.device, dtype=self.dtype))
        else:
            # Check if the keys are the same
            for key in vars(rhs):
                if not hasattr(self, key):
                    raise ValueError(f"Key '{key}' not found in lhs buffer.")
                lhs_val = getattr(self, key)
                rhs_val = getattr(rhs, key)
                if isinstance(lhs_val, torch.Tensor) and isinstance(rhs_val, torch.Tensor):
                    if lhs_val.shape[-1] != rhs_val.shape[-1]:
                        raise ValueError(
                            f"Dimensionality mismatch for key '{key}': {lhs_val.shape} vs {rhs_val.shape}."
                        )
                    setattr(self, key, torch.cat([lhs_val, rhs_val], dim=concat_dim))
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
            key: getattr(self.buffer, key)[idx]
            for key in vars(self.buffer)
        })

    def get_minibatch(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "PolicyBuffer":
        """
        Get a mini-batch of data from the buffer.
        """
        total_len = self.buffer.states.shape[0]
        indices = torch.randint(0, total_len, (batch_size,))
        batch = self.__getitem__(indices)
        return batch.sanitize_(device=device, dtype=dtype)
