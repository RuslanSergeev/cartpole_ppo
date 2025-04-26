import torch


def get_policy_loss(
    probability_ratio: torch.Tensor,
    advantages: torch.Tensor,
    epsilon: float = 0.2,
) -> torch.Tensor:
    """
    Calculate the policy loss for PPO.

    Args:
        log_probs (torch.Tensor): Current log probabilities of actions.
        advantages (torch.Tensor): Advantages for the current batch.
        old_log_probs (torch.Tensor): Old log probabilities of actions.
        epsilon (float): Clipping parameter.

    Returns:
        torch.Tensor: The calculated policy loss.
    """
    surrogate1 = probability_ratio * advantages
    surrogate2 = torch.clamp(
        probability_ratio, 1 - epsilon, 1 + epsilon
    ) * advantages
    return -torch.min(surrogate1, surrogate2).mean()


def get_entropy_loss(
    entropy: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate the entropy bonus for exploration.

    Args:
        entropy (torch.Tensor): The entropy of the action distribution.

    Returns:
        torch.Tensor: The calculated entropy loss.
    """
    return -entropy.mean()


def get_value_loss(
    values: torch.Tensor,
    returns: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate the value loss for PPO.

    Args:
        values (torch.Tensor): Current value estimates.
        returns (torch.Tensor): Target returns.

    Returns:
        torch.Tensor: The calculated value loss.
    """
    return ((values - returns) ** 2).mean()


def combine_losses(
    policy_loss: torch.Tensor,
    value_loss: torch.Tensor,
    entropy_loss: torch.Tensor,
    *,
    policy_weight: float = 1.0,
    value_weight: float = 1.0,
    entropy_weight: float = 0.01,
) -> torch.Tensor:
    """
    Calculate the total loss for PPO.

    Args:
        policy_loss (torch.Tensor): Policy loss.
        value_loss (torch.Tensor): Value loss.
        entropy_loss (torch.Tensor): Entropy loss.
        policy_weight (float): Weight for the policy loss.
        value_weight (float): Weight for the value loss.
        entropy_weight (float): Weight for the entropy loss.

    Returns:
        torch.Tensor: The total loss.
    """
    return (
        policy_weight * policy_loss +
        value_weight * value_loss +
        entropy_weight * entropy_loss
    )
