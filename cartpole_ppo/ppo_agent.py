import os
from typing import Union, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from .actions import (
    sample_normal_action,
    squash_action,
    squash_log_prob,
    get_probability_ratio,
)
from .return_estimators import (
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
from .policy_buffer import (
    PolicyBuffer,
    BufferDataset,
)
from .actor import Actor
from .critic import Critic
from .environment import InvertedPendulumEnv as Environment
from .logging import *
from .hardware_manager import Hardware_manager
from .checkpoints import Checkpoint


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
        buffer.to_tensor_(device=actor.device, dtype=value.dtype)
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
    return buffer.sanitize_(device=actor.device, dtype=value.dtype)


def validate_new_policy(
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
) -> torch.Tensor:
    """Utility function to get a random state for the cartpole environment."""
    x_init = np.random.uniform(x_init_min, x_init_max)
    theta_init = np.random.uniform(theta_init_min, theta_init_max)
    x_dot_init = np.random.uniform(x_dot_init_min, x_dot_init_max)
    theta_dot_init = np.random.uniform(
        theta_dot_init_min,
        theta_dot_init_max,
    )
    return torch.tensor([
        x_init,
        theta_init,
        x_dot_init,
        theta_dot_init,
    ])


def test_agent(
    actor: Actor,
    critic: Critic,
    env: Environment,
    episode: int,
    *,
    device: torch.device = torch.device("cpu"),
    num_time_steps: int = 500,
) -> None:
    """
    Test the agent in the environment.
    """
    # Set the initial state for each iteration
    state_init = torch.tensor(
        [
            0.0, # x position
            np.pi, # theta position
            0.0, # x velocity
            0.0, # theta velocity
        ],
        device=device,
        dtype=torch.float32,
    )
    # Rollout the old policy from state init
    with torch.no_grad():
        actor.eval()
        critic.eval()
        buffer = rollout_old_policy(
            state_init=state_init,
            environment=env,
            actor=actor,
            critic=critic,
            num_time_steps=num_time_steps,
        )
        # Print the collected total and average rewards
        total_reward = buffer.rewards.sum()
        average_reward = buffer.rewards.mean()
        logger.info(f"Episode {episode} summ reward: {total_reward.item()}")
        logger.info(f"Episode {episode} average reward: {average_reward.item()}")
        logger.info("__________________________________")


def train_cartpole_ppo(
    num_episodes: int = 5000,
    num_actors: int = 5,
    num_time_steps: int = 5000,
    num_epochs: int = 1000,
    *,
    lr_actor: float = 3e-4,
    lr_critic: float = 0.5e-3,
    gamma: float = 0.99,
    lam: float = 0.95,
    model_checkpoint_path: str = "model_checkpoint.pth",
    continue_training: bool = False,
    device: torch.device = torch.device("cpu"),
):
    """
    Run the PPO algorithm on the CartPole environment.
    """
    # Prepare the environment
    env = Environment(enable_rendering=False)
    # Actor network for policy approximation
    actor = Actor(state_dim=4, action_dim=1).to(device)
    # Critic network for value approximation
    critic = Critic(state_dim=4).to(device)
    # Optimizers:
    optimizer_actor = Adam(actor.parameters(), lr=lr_actor)
    optimizer_critic = Adam(critic.parameters(), lr=lr_critic)
    # scheduler for the actor
    scheduler_actor = StepLR(optimizer_actor, step_size=100, gamma=0.9)
    # scheduler for the critic
    scheduler_critic = StepLR(optimizer_critic, step_size=100, gamma=0.9)
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
    current_episode = checkpoint.data["episode"]
    for episode in range(current_episode, num_episodes):
        actor_old = Actor(state_dim=4, action_dim=1)
        actor_old.load_state_dict(actor.state_dict())
        actor_old.eval()
        critic_old = Critic(state_dim=4)
        critic_old.load_state_dict(critic.state_dict())
        critic_old.eval()
        # set up common buffer for all actors
        buffer = PolicyBuffer()
        # Rollout the old policy from state init
        with torch.no_grad():
            for _ in range(num_actors):
                # rollout the old policy
                current_buffer = rollout_old_policy(
                    state_init=get_random_state(),
                    environment=env,
                    actor=actor_old,
                    critic=critic_old,
                    num_time_steps=num_time_steps,
                    gamma=gamma,
                    lam=lam,
                )
                # add the current buffer to the common buffer
                buffer.add_buffer_(current_buffer)
        # Buffer-dataset:
        buffer_dataset = BufferDataset(buffer, permute=True)
        # optimize for num_epochs per iteration:
        for epoch in range(num_epochs):
            # update the actor and critic
            actor.train()
            critic.train()
            # get the new policy loss
            buffer_minibatch = buffer_dataset.get_minibatch(
                64, 
                device=device,
                dtype=torch.float32,
            )
            loss = validate_new_policy(
                actor=actor,
                critic=critic,
                buffer=buffer_minibatch,
            )
            # Perform optimization step
            optimizer_actor.zero_grad()
            optimizer_critic.zero_grad()
            loss.backward()
            optimizer_actor.step()
            optimizer_critic.step()
            # log the loss
            if epoch % 10 == 0:
                logger.info(f"Episode {episode}, Epoch {epoch}: Loss: {loss.item()}")
        # Test the agent
        test_agent(
            actor=actor,
            critic=critic,
            env=env,
            episode=episode,
            num_time_steps=num_time_steps,
        )
        # Update the learning rate
        scheduler_actor.step()
        scheduler_critic.step()
        # Save the trained models
        checkpoint.save(episode=episode)


def demo(checkpoint_path: str = "model_checkpoint.pth", num_time_steps: int = 5000):
    """
    Run a demo of the PPO agent on the CartPole environment.
    """
    # Load the actor and critic
    actor = Actor(state_dim=4, action_dim=1)
    critic = Critic(state_dim=4)
    environment = Environment(enable_rendering=True)
    # Load the checkpoint
    checkpoint = Checkpoint(
        checkpoint_path, 
        {
            "actor": actor,
            "critic": critic,
            "episode": 0,
        }
    )
    checkpoint.load()
    # Test the agent
    test_agent(
        actor=checkpoint.data["actor"],
        critic=checkpoint.data["critic"],
        env=environment,
        episode=checkpoint.data["episode"],
        num_time_steps=num_time_steps,
    )

if __name__ == "__main__":
    demo("model_checkpoint.pth")
#    train_cartpole_ppo(
#        model_checkpoint_path="model_checkpoint.pth",
#        num_episodes=5000,
#        num_actors=5,
#        num_epochs=50,
#        num_time_steps=3000,
#        device=Hardware_manager.get_device()
#    )
