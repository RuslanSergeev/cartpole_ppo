""" This is a Work In Progress replacement for the ppo_agent.py.
The original ppo_agent.py is written in procedural style.
This version is written in an object-oriented style, which 
makes the code more compact.
"""
from collections import defaultdict
from typing import Callable, Any
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


class PPO_agent:
    """
    Proximal Policy Optimization (PPO) agent.
    """
    def __init__(
        self,
        environment: Any,
        actor: nn.Module,
        critic: nn.Module,
        *,
        gamma: float = 0.99,
        lam: float = 0.95,
        epsilon: float = 0.2,
        train_init_state_generator: Callable,
        test_init_state_generator: Callable,
        device: torch.device = Hardware_manager.get_device(),
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize the PPO agent.
        
        Args:
        """
        self.environment = environment
        self.actor = actor
        self.critic = critic
        self.gamma = gamma
        self.lam = lam
        self.epsilon = epsilon

        self.train_init_state_generator = train_init_state_generator
        self.test_init_state_generator = test_init_state_generator
        self.device = device
        self.dtype = dtype

    def load(self, model_path: str):
        """
        Load the model from the checkpoint.
        """
        logger.info(f"Loading model from {model_path}")
        checkpoint = Checkpoint(
            model_path, 
            {
                "actor": self.actor,
                "critic": self.critic,
            }
        )
        checkpoint.load(self.device)
        logger.info("Model loaded successfully")

    def save(self, model_path):
        """
        Save the model to the checkpoint.
        """
        logger.info(f"Saving model to {model_path}")
        checkpoint = Checkpoint(
            model_path, 
            {
                "actor": self.actor,
                "critic": self.critic,
            }
        )
        checkpoint.save()
        logger.info("Model saved successfully")

    def rollout_old_policy(
        self,
        num_time_steps: int,
        init_state: np.ndarray
    ) -> RLDataset:
        """
        Rollout the old policy from state init.
        Args:
            num_time_steps (int): Number of time steps to rollout.
        Returns:
            RLDataset: Dataset containing the rollout data.
        """
        buffer = defaultdict(list)
        with torch.no_grad():
            self.actor.eval()
            self.critic.eval()
            # reset the environment
            state = self.environment.reset(init_state)
            for _ in tqdm(range(num_time_steps), desc="Rollout old policy", unit="step"):
                buffer["states"].append(state)
                # get the action distribution parameters
                action_mean, action_log_std = self.actor(state)
                # squeeze if the action is a scalar
                value = self.critic(state)
                # sample a normally distributed action
                action, log_prob, _ = sample_normal_action(
                    action_mean, action_log_std
                )
                # sample the environment
                state, reward, done = self.environment.step(
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
                    state = self.environment.reset()
            # criticize the last value
            value = self.critic(state)
            buffer["values"].append(value)

            dataset = RLDataset(device=self.device, dtype=self.dtype)
            dataset.add(**buffer)
            dataset.advantages = get_gae_advantages(
                dataset.rewards, 
                dataset.values, 
                dataset.dones, 
                gamma=self.gamma,
                lam=self.lam,
            )
            dataset.returns = get_gae_returns(dataset.advantages, dataset.values)
            dataset.advantages = normalize_advantages(dataset.advantages)
            dataset.detach()

        return dataset

    def validate_new_policy(
        self,
        *_,
        dataset: RLDataset,
        **__,
    ) -> torch.Tensor:
        """
        Evaluate the new policy using the old policy data.
        Given the states, actions, advantages, 
        """
        values = self.critic(dataset.states)
        action_mean, action_log_std = self.actor(dataset.states)
        _, __, entropy = sample_normal_action(action_mean, action_log_std)
        probability_ratio = get_probability_ratio(
            action_mean, action_log_std, dataset.actions, dataset.log_probs,
        )
        policy_loss = get_policy_loss(
            probability_ratio, dataset.advantages, self.epsilon
        )
        value_loss = get_value_loss(values, dataset.returns)
        entropy_loss = get_entropy_loss(entropy)
        loss = combine_losses(policy_loss, value_loss, entropy_loss)
        return loss

    def train_ppo(
        self,
        num_episodes: int = 5000,
        num_actors: int = 5,
        num_time_steps: int = 6000,
        num_epochs: int = 500,
        *,
        lr_actor: float = 1e-4,
        lr_critic: float = 0.5e-4,
        log_any: int = 100,
        batch_size: int = 32,
        model_checkpoint_path: str,
        continue_training: bool = False,
    ):
        """
        Run the PPO algorithm on the CartPole environment.
        """
        actor = self.actor.to(self.device)
        critic = self.critic.to(self.device)
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
            checkpoint.load(self.device)

        # Run num_iterations of rollouts and trainings
        first_episode = checkpoint.data["episode"]
        last_episode = first_episode + num_episodes
        for episode in range(first_episode, last_episode):
            # set up common buffer for all actors
            dataset = RLDataset(self.device, self.dtype)
            # Rollout the old policy from state init
            with torch.no_grad():
                for _ in range(num_actors):
                    # rollout the old policy
                    init_state = self.train_init_state_generator()
                    current_buffer = self.rollout_old_policy(
                        num_time_steps, init_state
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
                    loss = self.validate_new_policy(actor, critic, **batch)
                    # Perform optimization step
                    optimizer_actor.zero_grad()
                    optimizer_critic.zero_grad()
                    loss.backward()
                    optimizer_actor.step()
                    optimizer_critic.step()
                    # log the loss
                    if batch_idx % log_any == 0:
                        loss = loss.detach().cpu().item()
                        average_loss = loss / batch_size
                        logger.info(f"{episode}|{epoch}|{batch_idx}: Loss: {average_loss}")
            # Test the agent
            self.test_agent(
                num_time_steps=num_time_steps,
            )
            # Update the learning rate
            scheduler_actor.step()
            scheduler_critic.step()
            logger.info(f"lr_actor={scheduler_actor.get_last_lr()}" )
            logger.info(f"lr_critic={scheduler_critic.get_last_lr()}")
            # Save the trained models
            checkpoint.save(episode=episode)

    def test_agent(
        self,
        *,
        num_time_steps: int = 5000,
    ) -> None:
        """
        Test the agent in the environment.
        """
        # Rollout the old policy from initial state
        with torch.no_grad():
            self.actor.eval()
            self.critic.eval()
            init_state = self.test_init_state_generator()
            dataset = self.rollout_old_policy(num_time_steps, init_state)
            # Print the collected total and average rewards
            average_reward = dataset.rewards.mean().cpu().item()
            max_reward = dataset.rewards.max().cpu().item()
            logger.info(f"Average reward: {average_reward}")
            logger.info(f"Max reward: {max_reward}")
            logger.info("__________________________________")
