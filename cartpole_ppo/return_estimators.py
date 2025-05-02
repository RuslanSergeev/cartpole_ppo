import torch


def get_gae_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    ends: torch.Tensor,
    *,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> torch.Tensor:
    """
    Calculate the Generalized Advantage Estimation (GAE) for a batch of transitions.
    
    Args:
        rewards (torch.Tensor): Rewards for each transition.
        values (torch.Tensor): Value function estimates for each transition.
        ends (torch.Tensor): Mask indicating whether the episode has ended.
        gamma (float): Discount factor.
        lambda (float): Smoothing factor for GAE.
    
    Returns:
        advantages (torch.Tensor): The calculated advantages.
    """
    next_values = values[1:]
    values = values[:-1]
    
    masks = 1 - ends
    deltas = rewards + gamma * next_values * masks - values

    advantages = torch.zeros_like(rewards)
    rolling_advantage = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        rolling_advantage = deltas[t] + gamma * lam * rolling_advantage * masks[t]
        advantages[t] = rolling_advantage

    return advantages


def get_monte_carlo_returns(
    rewards: torch.Tensor,
    ends: torch.Tensor,
    *,
    gamma: float = 0.99,
) -> torch.Tensor:
    """
    Calculate the total returns for a batch of transitions.
    Markovian returns are calculated as:
    G_t = r_t + gamma * G_{t+1} * (1 - done_t)
    
    where G_t is the return at time t, r_t is the reward at time t,
    Args:
        rewards (torch.Tensor): Rewards for each transition.
        ends (torch.Tensor): Mask indicating whether the episode has ended.
        gamma (float): Discount factor.
    
    Returns:
        returns (torch.Tensor): The calculated total values.
    """
    masks = 1 - ends
    returns = torch.zeros_like(rewards)
    rolling_returns = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        rolling_return = rewards[t] + gamma * rolling_returns * masks[t]
        returns[t] = rolling_return

    return returns


def normalize_advantages(advantages: torch.Tensor) -> torch.Tensor:
    """
    Normalize the advantages to have mean 0 and standard deviation 1.
    
    Args:
        advantages (torch.Tensor): advantages for each transition.
    
    Returns:
        advantages (torch.Tensor): Normalized advantages .
    """
    mean = advantages .mean()
    std = advantages .std()
    normalized_advantages = (advantages - mean) / (std + 1e-8)

    return normalized_advantages


def get_bootstrap_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    ends: torch.Tensor,
    *,
    gamma: float = 0.99,
) -> torch.Tensor:
    """
    Calculate the bootstrap returns for a batch of transitions.
    Bootstrap returns are calculated as:
    G_t = r_t + gamma * V_{t+1} * (1 - done_t)
    
    where G_t is the return at time t, r_t is the reward at time t,
    V_{t+1} is the value estimate at time t+1, and done_t is a mask
    indicating whether the episode has ended.
    Args:
        rewards (np.ndarray): Rewards for each transition.
        values (np.ndarray): Value function estimates for each transition.
        ends (np.ndarray): Mask indicating whether the episode has ended.
        gamma (float): Discount factor.
    Returns:
        np.ndarray: The calculated bootstrap returns.
    """

    masks = 1 - ends
    next_values = values[1:]
    returns = rewards + gamma * next_values * masks

    return returns


def get_gae_returns(
    advantages: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate the advantage returns for a batch of transitions.
    Advantage returns are calculated as:
    G_t = A_t + V_t
    
    where G_t is the return at time t, A_t is the advantage at time t,
    and V_t is the value estimate at time t.
    Args:
        advantages (np.ndarray): Advantages for each transition.
        values (torch.Tensor): Value function estimates for each transition.
    Returns:
        returns (torch.Tensor): The calculated advantage returns. 
    """
    returns = advantages + values[:-1]
    return returns
