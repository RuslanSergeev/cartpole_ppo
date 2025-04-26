import torch
from torch import nn

class Critic(nn.Module):

    def __init__(self, state_dim: int, hidden_dim: int = 64):
        """
        Initialize the Critic network.

        Args:
            state_dim (int): Dimension of the input state.
            hidden_dim (int): Dimension of the hidden layers.
        """
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        Forward pass through the Critic network.

        Args:
            x (torch.Tensor): Input state tensor.

        Returns:
            torch.Tensor: Value estimate for the input state.
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value = self.value_head(x)
        return value
