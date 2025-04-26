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
    # Entropy of the distribution
    entropy = dist.entropy().sum(-1)
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
    dist = Normal(action_mean, torch.exp(action_log_std)) # Create a Normal distribution with mean and std
    norm_log_prob = dist.log_prob(action_normal).sum(-1) # Log probability of the sampled action
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
    corrected_log_prob = norm_log_prob - log_prob_correction
    return corrected_log_prob


def squash_action(
    norm_action: torch.Tensor,
    min_action: Union[float, torch.Tensor] = -1.0,
    max_action: Union[float, torch.Tensor] = 1.0,
) -> torch.Tensor:
    """
    Squash the action to the specified range.

    Args:
        norm_action (torch.Tensor): Action sampled from the Gaussian distribution, 
            range (-inf, inf).
        log_prob (torch.Tensor): The log probability of the normally sampled action.
            last dimension is squashed to 1.
        min_action (float): Minimum value for the action.
        max_action (float): Maximum value for the action.

    Returns:
        squashed_action (torch.Tensor): Squashed action, scaled to [min_action, max_action].
    """

    # Squash the action to [-1, 1]
    tanh_action = torch.tanh(norm_action) 

    # Scale to [min_action, max_action]
    delta_action = (max_action - min_action) / 2
    mean_action = (max_action + min_action) / 2
    squashed_action = mean_action + delta_action * tanh_action
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
