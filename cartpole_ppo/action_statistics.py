from typing import Tuple
import torch
from torch.distributions import Normal


def sample_normal_action(
    action_mean: torch.Tensor,
    action_log_std: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample an action from a Gaussian distribution with the given mean and log standard deviation.

    Args:
        mean_action (torch.Tensor): The mean of the Gaussian distribution.
        log_std (torch.Tensor): The log standard deviation of the Gaussian distribution.

    Returns:
        norm_action (torch.Tensor): Action sampled from the Gaussian distribution, 
            range (-inf, inf).
        log_prob (torch.Tensor): The log probability of the sampled action.
            last dimension is squashed to 1.
        entropy (torch.Tensor): The entropy of the action distribution.
    """
    # Saple the action from the Gaussian distribution, to [-1, 1]
    # Create a Normal distribution with mean and std
    dist = Normal(action_mean, torch.exp(action_log_std))
    # Sample from the distribution, range (-inf, inf)
    norm_action = dist.rsample()
    # Log probability of the sampled action
    log_prob = dist.log_prob(norm_action).sum(-1) # Sum the log prob over the last dimension
    log_prob.unsqueeze_(-1) # Add the squashed dimension
    entropy = dist.entropy().sum(-1) # Sum the entropy over the last dimension
    entropy.unsqueeze_(-1) # Add the squashed dimension
    return norm_action, log_prob, entropy


def get_norm_log_prob(
    action_mean: torch.Tensor,
    action_log_std: torch.Tensor,
    action_normal: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate the log probability of an action given the mean and log standard deviation
    of a Normally distributed action. Used to validate old actions on new distributions.
    Args:
        action_mean (torch.Tensor): The mean of the Gaussian distribution.
        action_log_std (torch.Tensor): The log standard deviation of the Gaussian distribution.
        action_normal (torch.Tensor): Action sampled from the Gaussian distribution, 
            range (-inf, inf).
    Returns:
    """
    std = torch.exp(action_log_std) # Convert log std to std
    dist = Normal(action_mean, std) # Create a Normal distribution with mean and std
    norm_log_prob = dist.log_prob(action_normal).sum(-1)
    norm_log_prob.unsqueeze_(-1) # Add the squashed dimension
    return norm_log_prob


def get_probability_ratio(
    new_action_mean: torch.Tensor,
    new_action_log_std: torch.Tensor,
    old_norm_action: torch.Tensor,
    old_norm_log_prob: torch.Tensor,
):
    """ Validate old actions on new distribution.
    All the old_* tensors are supposed to be detached.
    Args:
        new_action_mean (torch.Tensor): The mean of the new Gaussian distribution.
        new_action_log_std (torch.Tensor): The log standard deviation of the new Gaussian distribution.
        old_norm_action (torch.Tensor): Action sampled from the old Gaussian distribution, 
            range (-inf, inf).
        old_norm_log_prob (torch.Tensor): The log probability of the old action.
    """
    new_norm_log_prob = get_norm_log_prob(
        new_action_mean, 
        new_action_log_std, 
        old_norm_action
    )

    # Calculate the ratio of the new and old probabilities
    probability_ratio = torch.exp(new_norm_log_prob - old_norm_log_prob)
    return probability_ratio
