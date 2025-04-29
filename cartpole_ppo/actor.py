import torch
from torch import nn
from typing import overload

class Actor(nn.Module):
    """
    Actor network for PPO agent.
    """

    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        hidden_dim: int = 64,
        *,
        log_std_init: float = -0.6,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
    ):
        super(Actor, self).__init__()
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Parameter(
            torch.ones(action_dim) * log_std_init
        )

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mu = torch.tanh(self.mu_head(x))
        log_std = torch.clamp(
            self.log_std_head, 
            min=self.min_log_std, 
            max=self.max_log_std
        ).expand_as(mu)
        return mu, log_std

    @property
    def device(self):
        """
        Get the device of the model.
        """
        return self.fc1.weight.device
