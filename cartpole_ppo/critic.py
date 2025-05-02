import torch
from torch import nn
from .convert_to_tensor import convert_to_tensor

class Critic(nn.Module):

    def __init__(
        self,
        state_dim: int = 4,
        hidden_dim: int = 64
    ):
        """
        Initialize the Critic network.

        Args:
            state_dim (int): Dimension of the input state.
            hidden_dim (int): Dimension of the hidden layers.
        """
        super(Critic, self).__init__()
        self.state_dim = state_dim
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
        x = convert_to_tensor(
            x, 
            device=self.device, 
            dtype=self.dtype,
            feature_dim=self.state_dim
        )
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value = self.value_head(x)
        return value

    def clone(self):
        """
        Clone the Critic network.

        Returns:
            Critic: A new instance of the Critic network.
        """
        return Critic(
            state_dim=self.state_dim,
            hidden_dim=self.fc1.out_features
        ).to(self.device)

    @property
    def device(self):
        """
        Get the device of the model.
        """
        return self.fc1.weight.device

    @property
    def dtype(self):
        """
        Get the dtype of the model.
        """
        return self.fc1.weight.dtype
