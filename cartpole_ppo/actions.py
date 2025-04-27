from typing import Tuple, Union
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
    log_prob = get_norm_log_prob(action_mean, action_log_std, norm_action)
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
    of a Normally distributed action.
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


def squash_log_prob(
    norm_log_prob: torch.Tensor,
    tanh_action: torch.Tensor,
) -> torch.Tensor:
    """
    Correct the log probability of the action after squashing it to the range [-1, 1].
    Args:
        norm_log_prob (torch.Tensor): The log probability of the normally sampled action.
            last dimension is squashed to 1.
        tanh_action (torch.Tensor): Squashed action, scaled to [-1, 1].
    Returns:
        corrected_log_prob (torch.Tensor): The log probability of the squashed action.
    """
    # Correct the log probability of the action after squashing it to the range [-1, 1]
    log_prob_correction = torch.log(1 - tanh_action.pow(2) + 1e-6).sum(-1)
    log_prob_correction.unsqueeze_(-1) # Add the squashed dimension
    corrected_log_prob = norm_log_prob - log_prob_correction
    return corrected_log_prob


def squash_action(norm_action: torch.Tensor) -> torch.Tensor:
    """
    Squash the action to the specified range.

    Args:
        norm_action (torch.Tensor): Action sampled from the Gaussian distribution, 
            range (-inf, inf).
        log_prob (torch.Tensor): The log probability of the normally sampled action.
            last dimension is squashed to 1.
    Returns:
        squashed_action (torch.Tensor): Squashed action, scaled to (-1, 1).
    """

    # Squash the action to [-1, 1]
    squashed_action = torch.tanh(norm_action) 
    return squashed_action


def get_probability_ratio(
    new_action_mean: torch.Tensor,
    new_action_log_std: torch.Tensor,
    old_norm_action: torch.Tensor,
    old_tanh_action: torch.Tensor,
    old_squashed_log_prob: torch.Tensor,
):
    """ Validate old actions on new distribution.
    All the old_* tensors are supposed to be detached.
    Args:
        new_mean (torch.Tensor): The mean of the action of the new actor Gaussian distribution.
        new_log_std (torch.Tensor): The log standard deviation of the action of the new actor Gaussian distribution.
        old_norm_action (torch.Tensor): Action sampled from the old Gaussian distribution, 
            range (-inf, inf).
        old_tanh_action (torch.Tensor): Squashed action, scaled to [-1, 1].
        old_squashed_log_prob (torch.Tensor): The log probability of the squashed action.
    """
    new_norm_log_prob = get_norm_log_prob(
        new_action_mean, 
        new_action_log_std, 
        old_norm_action
    )
    new_squashed_log_prob = squash_log_prob(
        new_norm_log_prob, 
        old_tanh_action
    )

    # Calculate the ratio of the new and old probabilities
    probability_ratio = torch.exp(new_squashed_log_prob - old_squashed_log_prob)
    return probability_ratio
