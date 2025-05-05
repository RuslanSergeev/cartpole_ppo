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
        lr_actor: float = 1e-4,
        lr_critic: float = 0.5e-4,
        actor_scheduler_step_size: int = 20,
        actor_scheduler_gamma: float = 0.9,
        critic_scheduler_step_size: int = 20,
        critic_scheduler_gamma: float = 0.9,
        train_init_state_generator: Callable,
        test_init_state_generator: Callable,
        device: torch.device = Hardware_manager.get_device(),
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize the PPO agent.
        
        Args:
            environment (Any): The environment to train on.
                Should support the `reset` and `step` methods.
            actor (nn.Module): The actor network.
                Maps the state to the action distribution parameters.
            critic (nn.Module): The critic network.
                Maps the state to the value function.
            gamma (float): Discount factor for future rewards.
            lam (float): Lambda for GAE.
            epsilon (float): Epsilon for PPO.
            lr_actor (float): Learning rate for actor.
            lr_critic (float): Learning rate for critic.
            actor_scheduler_step_size (int): Step size for actor scheduler.
            actor_scheduler_gamma (float): Gamma for actor scheduler.
            critic_scheduler_step_size (int): Step size for critic scheduler.
            critic_scheduler_gamma (float): Gamma for critic scheduler.
            train_init_state_generator (Callable): Function to generate initial state for training.
            test_init_state_generator (Callable): Function to generate initial state for testing.
            device (torch.device): Device to run the model on.
            dtype (torch.dtype): Data type of the model parameters.
        """
        self.environment = environment
        self.actor = actor
        self.critic = critic
        self.gamma = gamma
        self.lam = lam
        self.epsilon = epsilon
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.actor_scheduler_step_size = actor_scheduler_step_size
        self.actor_scheduler_gamma = actor_scheduler_gamma
        self.critic_scheduler_step_size = critic_scheduler_step_size
        self.critic_scheduler_gamma = critic_scheduler_gamma

        self.train_init_state_generator = train_init_state_generator
        self.test_init_state_generator = test_init_state_generator
        self.device = device
        self.dtype = dtype

        # Set the actor and critic to the device
        self.actor = self.actor.to(self.device, dtype=self.dtype)
        self.critic = self.critic.to(self.device, dtype=self.dtype)

        # Optimizers, even if not used:
        self.optimizer_actor = Adam(self.actor.parameters(), lr=self.lr_actor)
        self.optimizer_critic = Adam(self.critic.parameters(), lr=self.lr_critic)
        # schedulers, even if not used:
        self.scheduler_actor = StepLR(
            self.optimizer_actor, 
            step_size=self.actor_scheduler_step_size,
            gamma=self.actor_scheduler_gamma
        )
        self.scheduler_critic = StepLR(
            self.optimizer_critic, 
            step_size=self.critic_scheduler_step_size,
            gamma=self.critic_scheduler_gamma
        )
        self.checkpoint_dict = {
            "actor": self.actor,
            "critic": self.critic,
            "actor_optimizer": self.optimizer_actor,
            "critic_optimizer": self.optimizer_critic,
            "actor_scheduler": self.scheduler_actor,
            "critic_scheduler": self.scheduler_critic,
            "episode": 0,
        }

    def load(self, checkpoint_path: str) -> None:
        """
        Load the model from a checkpoint.
        Args:
            checkpoint_path (str): Path to the checkpoint file.
        """
        Checkpoint.load(checkpoint_path, self.checkpoint_dict)

    def save(self, checkpoint_path: str) -> None:
        """
        Save the model to a checkpoint.
        Args:
            checkpoint_path (str): Path to the checkpoint file.
        """
        Checkpoint.save(checkpoint_path, self.checkpoint_dict)

    def _rollout(
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
                action, log_prob, _ = sample_normal_action(action_mean, action_log_std)
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

    def _validate(
        self,
        *_,
        states: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        **__,
    ) -> torch.Tensor:
        """
        Validate the `new` policy using the `old` policy data.
        Given the states, actions, advantages, 
        """
        values = self.critic(states)
        action_mean, action_log_std = self.actor(states)
        _, __, entropy = sample_normal_action(action_mean, action_log_std)
        probability_ratio = get_probability_ratio(
            action_mean, action_log_std, actions, log_probs,
        )
        policy_loss = get_policy_loss(
            probability_ratio, advantages, self.epsilon
        )
        value_loss = get_value_loss(values, returns)
        entropy_loss = get_entropy_loss(entropy)
        loss = combine_losses(policy_loss, value_loss, entropy_loss)
        return loss

    def train(
        self,
        num_episodes: int = 1000,
        num_actors: int = 1,
        num_time_steps: int = 6000,
        num_epochs: int = 10,
        *,
        log_any: int = 100,
        batch_size: int = 32,
        model_checkpoint_path: str,
    ):
        """
        Run the PPO algorithm on the CartPole environment.
        """
        actor = self.actor.to(self.device)
        critic = self.critic.to(self.device)
     
        # Run num_iterations of rollouts and trainings
        first_episode = self.checkpoint_dict["episode"]
        last_episode = first_episode + num_episodes
        for episode in range(first_episode, last_episode):
            # set up common buffer for all actors
            dataset = RLDataset(self.device, self.dtype)
            # Rollout the old policy from state init
            with torch.no_grad():
                for _ in range(num_actors):
                    # rollout the old policy
                    init_state = self.train_init_state_generator()
                    current_buffer = self._rollout(
                        num_time_steps, init_state
                    )
                    # add the current buffer to the common buffer
                    dataset.add(**current_buffer.data)

            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            # optimize for num_epochs per iteration:
            actor.train()
            critic.train()
            for epoch in range(num_epochs):
                for batch_idx, batch in enumerate(dataloader):
                    # get the new policy loss
                    loss = self._validate(**batch)
                    # Perform optimization step
                    self.optimizer_actor.zero_grad()
                    self.optimizer_critic.zero_grad()
                    loss.backward()
                    self.optimizer_actor.step()
                    self.optimizer_critic.step()
                    # log the loss
                    if batch_idx % log_any == 0:
                        loss = loss.detach().cpu().item()
                        average_loss = loss / len(batch)
                        logger.info(f"{episode}|{epoch}|{batch_idx}: Loss: {average_loss}")
            # Test the agent
            self.test(num_time_steps=num_time_steps)
            # Update the learning rate
            self.scheduler_actor.step()
            self.scheduler_critic.step()
            logger.info(f"lr_actor={self.scheduler_actor.get_last_lr()}" )
            logger.info(f"lr_critic={self.scheduler_critic.get_last_lr()}")
            # Save the trained models
            self.checkpoint_dict["episode"] = episode
            self.save(model_checkpoint_path)

    def test(self, *, num_time_steps: int = 5000) -> None:
        """
        Test the agent in the environment.
        """
        # Rollout the old policy from initial state
        with torch.no_grad():
            self.actor.eval()
            self.critic.eval()
            init_state = self.test_init_state_generator()
            dataset = self._rollout(num_time_steps, init_state)
            # Print the collected total and average rewards
            average_reward = dataset.rewards.mean().cpu().item()
            max_reward = dataset.rewards.max().cpu().item()
            logger.info(f"Average reward: {average_reward}")
            logger.info(f"Max reward: {max_reward}")
            logger.info("__________________________________")
