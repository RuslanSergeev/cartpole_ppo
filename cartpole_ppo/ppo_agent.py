from typing import Union
import torch
import torch.nn as nn

from .actions import (
    sample_normal_action,
    squash_action,
    squash_log_prob,
    get_probability_ratio,
)
from .gae_advantage import (
    get_gae_advantages,
    get_gae_returns,
)
from .loss_functions import (
    get_policy_loss,
    get_entropy_loss,
    get_value_loss,
    combine_losses,
)


class PolicyBuffer:
    """
    Buffer to store policy data for PPO.
    """

    def __init__(self):
        self.states = []
        self.actions_normal = []
        self.actions_squashed = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.returns = []
        self.advantages = []
        self.dones = []
        self.is_tensor = False

    def to_tensors(self):
        """
        Convert the buffer data to tensors.
        """
        if not self.is_tensor:
            self.states = torch.stack(self.states).detach()
            self.actions_normal = torch.stack(self.actions_normal).detach()
            self.actions_squashed = torch.stack(self.actions_squashed).detach()
            self.log_probs = torch.stack(self.log_probs).detach()
            self.rewards = torch.stack(self.rewards).detach()
            self.values = torch.stack(self.values).detach()
            self.dones = torch.stack(self.dones).detach()
        return self


def rollout_old_policy(
    state_init: torch.Tensor,
    environment,
    actor: nn.Module,
    critic: nn.Module,
    num_time_steps: int,
    *,
    action_min: Union[float, torch.Tensor] = -1.0,
    action_max: Union[float, torch.Tensor] = 1.0,
    gamma: float = 0.99,
    lam: float = 0.95,
):
    buffer = PolicyBuffer()
    actor.eval()
    critic.eval()
    # reset the environment
    state = environment.reset(state_init.cpu().numpy())
    state = torch.tensor(state, device=actor.device, dtype=state_init.dtype)
    for _ in range(num_time_steps):
        buffer.states.append(state)

        # get the action distribution parameters
        action_mean, action_log_std = actor(state)
        value = critic(state)
        # sample a normally distributed action
        normal_action, normal_log_prob, _ = sample_normal_action(
            action_mean.detach(),
            action_log_std.detach(),
        )
        # squash to an action, the environment can understand
        squashed_action = squash_action(
            normal_action,
            action_min,
            action_max,
        )
        # correct the log probability
        squashed_log_prob = squash_log_prob(
            normal_log_prob,
            squashed_action,
        )
        # sample the environment
        state, reward, done = environment.step(
            squashed_action.detach().cpu().numpy()
        )
        state = torch.tensor(state, device=actor.device, dtype=value.dtype)
        reward = torch.tensor(reward, device=actor.device, dtype=value.dtype)
        done = torch.tensor(done, device=actor.device, dtype=value.dtype)
        # buffer the old policy data
        buffer.actions_normal.append(normal_action)
        buffer.actions_squashed.append(squashed_action)
        buffer.log_probs.append(squashed_log_prob)
        buffer.rewards.append(reward)
        buffer.values.append(value)
        buffer.dones.append(done)
    buffer.to_tensors()
    # criticize the last value
    value = critic(state)
    buffer.values.append(value)
    # calculate the advantages
    buffer.advantages = get_gae_advantages(
        buffer.rewards,
        buffer.values,
        buffer.dones,
        gamma=gamma,
        lam=lam,
    )
    buffer.returns = get_gae_returns(
        buffer.advantages,
        buffer.values,
    )
    return buffer.to_tensors()


def evaluate_new_policy(
    actor: nn.Module,
    critic: nn.Module,
    buffer: PolicyBuffer,
    *,
    epsilon: float = 0.2,
) -> torch.Tensor:
    """
    Evaluate the new policy using the old policy data.
    Obtains the policy probability ratios, gae advantages, and critic values.
    """
    values = critic(buffer.states)
    action_mean, action_log_std = actor(buffer.states)
    _, __, entropy = sample_normal_action(
        action_mean,
        action_log_std,
    )
    probability_ratio = get_probability_ratio(
        action_mean,
        action_log_std,
        buffer.actions_normal,
        buffer.actions_squashed,
        buffer.log_probs,
    )
    # calculate the policy loss
    policy_loss = get_policy_loss(
        probability_ratio,
        buffer.advantages,
        epsilon=epsilon,
    )
    # calculate the value loss
    value_loss = get_value_loss(
        values,
        buffer.returns,
    )
    # calculate the entropy loss
    entropy_loss = get_entropy_loss(entropy)
    # combine the losses
    loss = combine_losses(
        policy_loss,
        value_loss,
        entropy_loss,
    )
    return loss
