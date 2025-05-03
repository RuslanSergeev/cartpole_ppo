import torch
from torch import nn
from .convert_to_tensor import convert_to_tensor

class Actor(nn.Module):
    """
    Actor network for PPO agent.
    """

    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 1,
        hidden_dim: int = 64,
        *,
        action_bias: float = 0.0,
        action_scale: float = 3.0,
        log_std_init: float = -2.0,
        min_log_std: float = -20.0,
        max_log_std: float = -1.0,
    ):
        super(Actor, self).__init__()
        # Copy the arguments
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.action_bias = action_bias
        self.action_scale = action_scale
        self.log_std_init = log_std_init
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        # Initialize the networks
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Parameter(
            torch.ones(action_dim) * log_std_init
        )

    def forward(self, x):
        # convert to tensor if needed
        x = convert_to_tensor(
            x, 
            device=self.device, 
            dtype=self.dtype, 
            feature_dim=self.state_dim
        )
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mu = torch.tanh(self.mu_head(x))
        mu = mu * self.action_scale + self.action_bias
        log_std = torch.clamp(
            self.log_std_head, 
            min=self.min_log_std, 
            max=self.max_log_std
        ).expand_as(mu)
        return mu, log_std

    def clone(self):
        """
        Clone the actor network.
        """
        clone = Actor(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            action_bias=self.action_bias,
            action_scale=self.action_scale,
            log_std_init=self.log_std_init,
            min_log_std=self.min_log_std,
            max_log_std=self.max_log_std
        ).to(self.device)
        clone.load_state_dict(self.state_dict().copy())

        return clone

    @property
    def device(self):
        """
        Get the device of the model.
        """
        return self.parameters().__next__().device

    @property
    def dtype(self):
        """
        Get the dtype of the model.
        """
        return self.parameters().__next__().dtype
