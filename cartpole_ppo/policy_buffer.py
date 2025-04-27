from typing import Optional
import torch

class PolicyBuffer:
    """
    Buffer to store policy data for PPO.
    """

    def __init__(self):
        self.states = []
        self.actions_normal = []
        self.actions_squashed = []
        self.log_probs_squashed = []
        self.rewards = []
        self.values = []
        self.returns = []
        self.advantages = []
        self.dones = []

    def to_tensor(
        self,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Convert the buffer data to tensors.
        """
        for key in self.__dict__:
            val = self.__dict__[key]
            if isinstance(val, list) and len(val):
                val = torch.stack(val)
                self.__dict__[key] = val
            if isinstance(val, torch.Tensor):
                val = val.to(
                    device=device,
                    dtype=dtype,
                ).detach()
                self.__dict__[key] = val
        return self
