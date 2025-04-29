import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

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
from .rl_dataset import RLDataset
from .actor import Actor
from .critic import Critic
from .environment import InvertedPendulumEnv as Environment
from .logging import logger
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
) -> RLDataset:
    """
    Rollout the old policy from state init.
    Args:
        state_init (torch.Tensor): Initial state of the environment.
        environment: Environment to sample from.
        actor (nn.Module): Actor network for policy approximation.
        critic (nn.Module): Critic network for value approximation.
        num_time_steps (int): Number of time steps to rollout.
        gamma (float): Discount factor.
        lam (float): Lambda for GAE.
    Returns:
        RLDataset: Dataset containing the rollout data.
    """
    actor.eval()
    critic.eval()
    # Create the buffers:
    buffer = {
        "states": [],
        "actions_normal": [],
        "actions_squashed": [],
        "log_probs_squashed": [],
        "rewards": [],
        "values": [],
        "dones": [],
    }
    with torch.no_grad():
        # reset the environment
        state = environment.reset(state_init.cpu().numpy())
        for _ in range(num_time_steps):
            buffer["states"].append(state)
            # get the action distribution parameters
            tensor_state = torch.Tensor(
                state, device=actor.device
            ).unsqueeze(0)
            action_mean, action_log_std = actor(tensor_state)
            # squeeze if the action is a scalar
            value = critic(tensor_state)
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
            # buffer the old policy data
            buffer["actions_normal"].append(action_normal)
            buffer["actions_squashed"].append(action_squashed)
            buffer["log_probs_squashed"].append(log_prob_squashed)
            buffer["rewards"].append(reward)
            buffer["values"].append(value)
            buffer["dones"].append(done)
        # criticize the last value
        value = critic(state)
        buffer["values"].append(value)

        dataset = RLDataset(device=actor.device, dtype=value.dtype)
        dataset.add(**buffer)
        dataset.rewards = normalize_rewards(dataset.rewards)
        dataset.advantages = get_gae_advantages(
            dataset.rewards,
            dataset.values,
            dataset.dones,
            gamma=gamma,
            lam=lam,
        )
        dataset.returns = get_gae_returns(dataset.advantages, dataset.values)
    return dataset


def validate_new_policy(
    actor: nn.Module,
    critic: nn.Module,
    *,
    epsilon: float = 0.2,
    states: torch.Tensor,
    actions_normal: torch.Tensor,
    actions_squashed: torch.Tensor,
    log_probs_squashed: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
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
        actions_normal,
        actions_squashed,
        log_probs_squashed,
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
    num_time_steps: int = 500,
) -> None:
    """
    Test the agent in the environment.
    """
    # Set the initial state for each iteration
    qpos_qvel_init = [0.0, np.pi, 0.0, 0.0],
    state_init = torch.tensor(
        qpos_qvel_init, device=actor.device, dtype=torch.float32
    )

    # Rollout the old policy from state init
    with torch.no_grad():
        actor.eval()
        critic.eval()
        dataset = rollout_old_policy(
            state_init=state_init,
            environment=env,
            actor=actor,
            critic=critic,
            num_time_steps=num_time_steps,
        )
        # Print the collected total and average rewards
        total_reward = dataset.rewards.sum()
        average_reward = dataset.rewards.mean()
        lr_actor = actor.optimizer.param_groups[0]["lr"]
        lr_critic = critic.optimizer.param_groups[0]["lr"]
        logger.info(f"Episode {episode} actor lr: {lr_actor}")
        logger.info(f"Episode {episode} critic lr: {lr_critic}")
        logger.info(f"Episode {episode} summ reward: {total_reward.item()}")
        logger.info(f"Episode {episode} average reward: {average_reward.item()}")
        logger.info("__________________________________")


def train_cartpole_ppo(
    num_episodes: int = 5000,
    num_actors: int = 5,
    num_time_steps: int = 6000,
    num_epochs: int = 500,
    *,
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
        dataset = RLDataset(device=device, dtype=torch.float32)
        # Rollout the old policy from state init
        qpos_qvel_init = [0.0, np.pi, 0.0, 0.0],
        state_init=torch.tensor(
            qpos_qvel_init, device=device, dtype=torch.float32
        )

        with torch.no_grad():
            for _ in range(num_actors):
                # rollout the old policy
                current_buffer = rollout_old_policy(
                    state_init=state_init,
                    environment=env,
                    actor=actor_old,
                    critic=critic_old,
                    num_time_steps=num_time_steps,
                    gamma=gamma,
                    lam=lam,
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
                    logger.info(f"{episode}/{epoch}/{batch_idx}: Loss: {loss.item()}")
        # Test the agent
        test_agent(actor, critic, env, episode,
            num_time_steps=num_time_steps,
        )
        # Update the learning rate
        scheduler_actor.step()
        scheduler_critic.step()
        # Save the trained models
        checkpoint.save(episode=episode)


def demo_cartpole_ppo(checkpoint_path: str = "model_checkpoint.pth", num_time_steps: int = 5000):
    """
    Run a demo of the PPO agent on the CartPole environment.
    """
    # Load the actor and critic
    actor = Actor(state_dim=4, action_dim=1).to(device=Hardware_manager.get_device())
    critic = Critic(state_dim=4).to(device=Hardware_manager.get_device())
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
