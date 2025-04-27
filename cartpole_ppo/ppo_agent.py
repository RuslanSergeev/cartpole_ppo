from typing import Union, Optional
import numpy as np
import torch
import torch.nn as nn

from .actions import (
    sample_normal_action,
    squash_action,
    squash_log_prob,
    get_probability_ratio,
)
from .returns import (
    get_gae_advantages,
    get_gae_returns,
    normalize_rewards,
)
from .loss_functions import (
    get_policy_loss,
    get_entropy_loss,
    get_value_loss,
    combine_losses,
)
from .actor import Actor
from .critic import Critic
from .environment import InvertedPendulumEnv as Environment
from .policy_buffer import PolicyBuffer
from .random_init import set_seed
from .logging import *

def rollout_old_policy(
    state_init: torch.Tensor,
    environment,
    actor: nn.Module,
    critic: nn.Module,
    num_time_steps: int,
    *,
    gamma: float = 0.99,
    lam: float = 0.95,
):
    buffer = PolicyBuffer()
    actor.eval()
    critic.eval()
    with torch.no_grad():
        # reset the environment
        state = environment.reset(state_init.cpu().numpy())
        state = torch.tensor(state, device=actor.device, dtype=state_init.dtype)
        state = torch.atleast_1d(state)
        for _ in range(num_time_steps):
            buffer.states.append(state)
            # get the action distribution parameters
            action_mean, action_log_std = actor(state)
            # squeeze if the action is a scalar
            value = critic(state)
            # sample a normally distributed action
            action_normal, log_prob_normal, _ = sample_normal_action(
                action_mean,
                action_log_std,
            )
            # squash to an action, the environment can understand
            action_squashed = squash_action(action_normal)
            # correct the log probability
            log_prob_squashed = squash_log_prob(
                log_prob_normal,
                action_squashed,
            )
            # sample the environment
            state, reward, done = environment.step(
                action_squashed.cpu().numpy()
            )
            # Transform to tensors and ensure they are 1D
            state = torch.tensor(state, device=actor.device, dtype=value.dtype)
            state = torch.atleast_1d(state)
            reward = torch.tensor(reward, device=actor.device, dtype=value.dtype)
            reward = torch.atleast_1d(reward)
            done = torch.tensor(done, device=actor.device, dtype=value.dtype)
            done = torch.atleast_1d(done)
            # buffer the old policy data
            buffer.actions_normal.append(action_normal)
            buffer.actions_squashed.append(action_squashed)
            buffer.log_probs_squashed.append(log_prob_squashed)
            buffer.rewards.append(reward)
            buffer.values.append(value)
            buffer.dones.append(done)
        # criticize the last value
        value = critic(state)
        buffer.values.append(value)
        # calculate the advantages
        buffer.to_tensor(device=actor.device, dtype=value.dtype)
        # normalize the rewards
        buffer.rewards = normalize_rewards(buffer.rewards)
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
    return buffer.to_tensor(device=actor.device, dtype=value.dtype)


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
        buffer.log_probs_squashed,
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


def get_random_state(
    x_init_min: float = -1.0,
    x_init_max: float = 1.0,
    theta_init_min: float = -np.pi,
    theta_init_max: float = np.pi,
    x_dot_init_min: float = -1.0,
    x_dot_init_max: float = 1.0,
    theta_dot_init_min: float = -1.0,
    theta_dot_init_max: float = 1.0,
) -> tuple:
    """Utility function to get a random state for the cartpole environment."""
    x_init = np.random.uniform(x_init_min, x_init_max)
    theta_init = np.random.uniform(theta_init_min, theta_init_max)
    x_dot_init = np.random.uniform(x_dot_init_min, x_dot_init_max)
    theta_dot_init = np.random.uniform(
        theta_dot_init_min,
        theta_dot_init_max,
    )
    return (
        x_init,
        theta_init,
        x_dot_init,
        theta_dot_init,
    )


def test_agent(
    actor: nn.Module = None,
    critic: nn.Module = None,
    num_time_steps: int = 500,
    episode: int = 0,
) -> None:
    """
    Test the agent in the environment.
    """
    # Prepare the environment
    env = Environment(enable_rendering=False)
    # Set the initial state for each iteration
    state_init = torch.tensor(
        [
            0.0, # x position
            np.pi, # theta position
            0.0, # x velocity
            0.0, # theta velocity
        ],
        device=actor.device,
        dtype=torch.float32,
    )
    # Rollout the old policy from state init
    with torch.no_grad():
        buffer = rollout_old_policy(
            state_init=state_init,
            environment=env,
            actor=actor,
            critic=critic,
            num_time_steps=num_time_steps,
        )
        # Print the collected total and average rewards
        total_reward = sum(buffer.rewards)
        average_reward = total_reward / num_time_steps
        logger.info(f"Episode {episode} summ reward: {total_reward.item()}")
        logger.info(f"Episode {episode} average reward: {average_reward.item()}")
        logger.info("__________________________________")


def train_cartpole_ppo(
    num_episodes: int = 5000,
    num_time_steps: int = 5000,
    num_epochs: int = 500,
    *,
    lr_actor: float = 3e-4,
    lr_critic: float = 3e-3,
    gamma: float = 0.99,
    lam: float = 0.95,
    seed: int = 42,
    device: Optional[torch.device] = None,
):
    """
    Run the PPO algorithm on the CartPole environment.
    """
    # Set the random seed
    set_seed(seed)
    # Prepare the environment
    env = Environment(enable_rendering=False)
    # Actor network for policy approximation
    actor = Actor(state_dim=4, action_dim=1).to(device)
    # Critic network for value approximation
    critic = Critic(state_dim=4).to(device)
    # Optimizers:
    optimizer_actor = torch.optim.Adam(
        actor.parameters(),
        lr=lr_actor,
    )
    optimizer_critic = torch.optim.Adam(
        critic.parameters(),
        lr=lr_critic,
    )
    # scheduler for the actor
    scheduler_actor = torch.optim.lr_scheduler.StepLR(
        optimizer_actor,
        step_size=1000,
        gamma=0.1,
    )
    # scheduler for the critic
    scheduler_critic = torch.optim.lr_scheduler.StepLR(
        optimizer_critic,
        step_size=1000,
        gamma=0.1,
    )
    # Run num_iterations of rollouts and trainings 
    for episode in range(num_episodes):
        actor_old = Actor(state_dim=4, action_dim=1)
        actor_old.load_state_dict(actor.state_dict())
        actor_old.eval()
        critic_old = Critic(state_dim=4)
        critic_old.load_state_dict(critic.state_dict())
        critic_old.eval()
        # Set the initial state for each iteration
        state_init = torch.tensor(
            get_random_state(),
            device=actor.device,
            dtype=torch.float32,
        )
        # Rollout the old policy from state init
        with torch.no_grad():
            buffer = rollout_old_policy(
                state_init=state_init,
                environment=env,
                actor=actor_old,
                critic=critic_old,
                num_time_steps=num_time_steps,
                gamma=gamma,
                lam=lam,
            )
        # optimize for num_epochs per iteration:
        for epoch in range(num_epochs):
            # update the actor and critic
            actor.train()
            critic.train()
            # get the new policy loss
            loss = evaluate_new_policy(
                actor=actor,
                critic=critic,
                buffer=buffer,
            )
            # Perform optimization step
            optimizer_actor.zero_grad()
            optimizer_critic.zero_grad()
            loss.backward()
            optimizer_actor.step()
            optimizer_critic.step()
            # update the learning rate
            scheduler_actor.step()
            scheduler_critic.step()
            # log the loss
            if epoch % 10 == 0:
                logger.info(f"Episode {episode}, Epoch {epoch}: Loss: {loss.item()}")
        # Test the agent
        test_agent(
            actor=actor,
            critic=critic,
            num_time_steps=num_time_steps,
        )
        # Save the trained models
        logger.info("Saving the models...")
        torch.save(actor.state_dict(), "actor.pth")
        torch.save(critic.state_dict(), "critic.pth")
    # Test the trained agent
    logger.info("Final test of the agent:")
    test_agent(
        actor=actor,
        critic=critic,
        num_time_steps=num_time_steps,
        episode=num_episodes,
    )


if __name__ == "__main__":
    train_cartpole_ppo()
