import torch
from torch import nn

class Actor(nn.Module):
    """
    Actor network for PPO agent.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_std = self.log_std_head
        return mu, log_std
