from typing import Optional, List
import torch

class PolicyBuffer:
    """
    Buffer to store policy data for PPO.
    """

    # Which keys are only used during rollout and should be removed
    rollout_only_keys = {"rewards", "values", "dones"}

    def __init__(self):
        self.states: List[torch.Tensor] = []
        self.actions_normal: List[torch.Tensor] = []
        self.actions_squashed: List[torch.Tensor] = []
        self.log_probs_squashed: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.returns: List[torch.Tensor] = []
        self.advantages: List[torch.Tensor] = []
        self.dones: List[torch.Tensor] = []

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
