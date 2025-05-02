from collections import defaultdict
from typing import Callable, Any
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from .action_statistics import (
    sample_normal_action,
    get_probability_ratio,
)
from .return_estimators import (
    get_gae_advantages,
    get_gae_returns,
    normalize_advantages,
)
from .loss_functions import (
    get_policy_loss,
    get_entropy_loss,
    get_value_loss,
    combine_losses,
)
from .rl_dataset import RLDataset
from .hardware_manager import Hardware_manager
from .model_checkpoints import Checkpoint
from .logging import logger


def rollout_old_policy(
    environment,
    actor: nn.Module,
    critic: nn.Module,
    num_time_steps: int,
    *,
    gamma: float = 0.99,
    lam: float = 0.95,
    device: torch.device = Hardware_manager.get_device(),
) -> RLDataset:
    """
    Rollout the old policy from state init.
    Args:
        state_init (np.ndarray): Initial state of the environment.
        environment: Environment to sample from.
        actor (nn.Module): Actor network for policy approximation.
        critic (nn.Module): Critic network for value approximation.
        num_time_steps (int): Number of time steps to rollout.
        gamma (float): Discount factor.
        lam (float): Lambda for GAE.
    Returns:
        RLDataset: Dataset containing the rollout data.
    """
    actor = actor.to(device).eval()
    critic = critic.to(device).eval()
    # Create the buffers:
    buffer = defaultdict(list)
    with torch.no_grad():
        # reset the environment
        state = environment.reset()
        for _ in tqdm(range(num_time_steps), desc="Rollout old policy", unit="step"):
            buffer["states"].append(state)
            # get the action distribution parameters
            action_mean, action_log_std = actor(state)
            # squeeze if the action is a scalar
            value = critic(state)
            # sample a normally distributed action
            action, log_prob, _ = sample_normal_action(
                action_mean,
                action_log_std,
            )
            # sample the environment
            state, reward, done = environment.step(
                action.cpu().numpy()
            )
            # buffer the old policy data
            buffer["actions"].append(action)
            buffer["log_probs"].append(log_prob)
            buffer["rewards"].append(reward)
            buffer["values"].append(value)
            buffer["dones"].append(done)
            # if the episode is done, reset the environment
            if done:
                state = environment.reset()
        # criticize the last value
        value = critic(state)
        buffer["values"].append(value)

        dataset = RLDataset(device=actor.device, dtype=value.dtype)
        dataset.add(**buffer)
        dataset.advantages = get_gae_advantages(
            dataset.rewards,
            dataset.values,
            dataset.dones,
            gamma=gamma,
            lam=lam,
        )
        dataset.returns = get_gae_returns(
            dataset.advantages, 
            dataset.values
        )
        dataset.advantages = normalize_advantages(
            dataset.advantages
        )
    dataset.detach()

    return dataset


def validate_new_policy(
    actor: nn.Module,
    critic: nn.Module,
    *_,
    epsilon: float = 0.2,
    states: torch.Tensor,
    actions: torch.Tensor,
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    **__,
) -> torch.Tensor:
    """
    Evaluate the new policy using the old policy data.
    Obtains the policy probability ratios, gae advantages, and critic values.
    """
    values = critic(states)
    action_mean, action_log_std = actor(states)
    _, __, entropy = sample_normal_action(action_mean, action_log_std)
    probability_ratio = get_probability_ratio(
        action_mean,
        action_log_std,
        actions,
        log_probs,
    )
    policy_loss = get_policy_loss(probability_ratio, advantages, epsilon)
    value_loss = get_value_loss(values, returns)
    entropy_loss = get_entropy_loss(entropy)
    loss = combine_losses(
        policy_loss,
        value_loss,
        entropy_loss,
    )
    return loss


def test_agent(
    actor: nn.Module,
    critic: nn.Module,
    env: Any,
    *,
    num_time_steps: int = 5000,
) -> None:
    """
    Test the agent in the environment.
    """
    # Rollout the old policy from initial state
    with torch.no_grad():
        actor.eval()
        critic.eval()
        dataset = rollout_old_policy(
            environment=env,
            actor=actor,
            critic=critic,
            num_time_steps=num_time_steps,
        )
        # Print the collected total and average rewards
        total_reward = dataset.rewards.sum().cpu().item()
        average_reward = dataset.rewards.mean().cpu().item()
        max_reward = dataset.rewards.max().cpu().item()
        logger.info(f"Summ reward: {total_reward}")
        logger.info(f"Average reward: {average_reward}")
        logger.info(f"Max reward: {max_reward}")
        logger.info("__________________________________")


def train_ppo(
    num_episodes: int = 5000,
    num_actors: int = 5,
    num_time_steps: int = 6000,
    num_epochs: int = 500,
    *,
    environment,
    actor: nn.Module,
    critic: nn.Module,
    lr_actor: float = 1e-4,
    lr_critic: float = 0.5e-4,
    gamma: float = 0.99,
    lam: float = 0.95,
    log_any: int = 100,
    batch_size: int = 32,
    model_checkpoint_path: str = "model_checkpoint.pth",
    continue_training: bool = False,
    device: torch.device = Hardware_manager.get_device(),
):
    """
    Run the PPO algorithm on the CartPole environment.
    """
    actor = actor.to(device)
    critic = critic.to(device)
    actor_old = actor.clone().eval().to(device)
    critic_old = critic.clone().eval().to(device)
    # Optimizers:
    optimizer_actor = Adam(actor.parameters(), lr=lr_actor)
    optimizer_critic = Adam(critic.parameters(), lr=lr_critic)
    # scheduler for the actor
    scheduler_actor = StepLR(optimizer_actor, step_size=20, gamma=0.9)
    # scheduler for the critic
    scheduler_critic = StepLR(optimizer_critic, step_size=20, gamma=0.9)
    # Checkpoint for saving the model
    checkpoint = Checkpoint(
        model_checkpoint_path, 
        {
            "episode": 0,
            "actor": actor,
            "critic": critic,
            "actor_optimizer": optimizer_actor,
            "critic_optimizer": optimizer_critic,
            "actor_scheduler": scheduler_actor,
            "critic_scheduler": scheduler_critic,
        }
    )
    if continue_training:
        checkpoint.load()

    # Run num_iterations of rollouts and trainings
    first_episode = checkpoint.data["episode"]
    last_episode = first_episode + num_episodes
    for episode in range(first_episode, last_episode):
        actor_old.load_state_dict(actor.state_dict())
        critic_old.load_state_dict(critic.state_dict())
        # set up common buffer for all actors
        dataset = RLDataset(device=device, dtype=actor.dtype)
        # Rollout the old policy from state init
        with torch.no_grad():
            for _ in range(num_actors):
                # rollout the old policy
                current_buffer = rollout_old_policy(
                    environment=environment,
                    actor=actor_old,
                    critic=critic_old,
                    num_time_steps=num_time_steps,
                    gamma=gamma,
                    lam=lam,
                    device=device,
                )
                # add the current buffer to the common buffer
                dataset.add(**current_buffer.data)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        # optimize for num_epochs per iteration:
        for epoch in range(num_epochs):
            for batch_idx, batch in enumerate(dataloader):
                actor.train()
                critic.train()
                # get the new policy loss
                loss = validate_new_policy(actor, critic, **batch)
                # Perform optimization step
                optimizer_actor.zero_grad()
                optimizer_critic.zero_grad()
                loss.backward()
                optimizer_actor.step()
                optimizer_critic.step()
                # log the loss
                if batch_idx % log_any == 0:
                    loss = loss.detach().cpu().item()
                    logger.info(f"{episode}/{epoch}/{batch_idx}: Loss: {loss}")
        # Test the agent
        test_agent(
            actor, critic, environment, num_time_steps=num_time_steps,
        )
        # Update the learning rate
        scheduler_actor.step()
        scheduler_critic.step()
        logger.info(f"lr_actor={scheduler_actor.get_last_lr()}" )
        logger.info(f"lr_critic={scheduler_critic.get_last_lr()}")
        # Save the trained models
        checkpoint.save(episode=episode)
