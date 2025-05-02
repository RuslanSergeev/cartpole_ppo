import sys

from cartpole_ppo.ppo_agent import train_ppo
from cartpole_ppo.actor import Actor
from cartpole_ppo.critic import Critic
from cartpole_ppo.environment import InvertedPendulumEnv as Environment
from cartpole_ppo.hardware_manager import Hardware_manager
from cartpole_ppo.environment import demo_cartpole_ppo
from cartpole_ppo.model_checkpoints import Checkpoint
from cartpole_ppo.state_generators import (
    get_pendulum_down_state,
    get_random_state,
)


def train(
    checkpoint_path: str,
    device = Hardware_manager.get_device(),
):
    # Prepare the environment
    environment = Environment(
        enable_rendering=False,
        initial_state_generator=get_random_state,
    )
    # Actor network for policy approximation
    actor = Actor(state_dim=4, action_dim=1).to(device)
    # Critic network for value approximation
    critic = Critic(state_dim=4).to(device)

    train_ppo(
        model_checkpoint_path=checkpoint_path,
        environment=environment,
        actor=actor,
        critic=critic,
        lr_actor=1e-5,
        lr_critic=0.5e-4,
        num_episodes=5000,
        num_actors=20,
        num_epochs=10,
        num_time_steps=6000,
        batch_size=64,
        log_any=10,
        device=Hardware_manager.get_device()
    )


def demo(
    checkpoint_path: str,
    device = Hardware_manager.get_device()
):
    actor = Actor(state_dim=4, action_dim=1).to(device)
    critic = Critic(state_dim=4).to(device)
    # Load the trained model
    checkpoint = Checkpoint(
        checkpoint_path, 
        {
            "episode": 0,
            "actor": actor,
            "critic": critic,
        }
    )
    checkpoint.load()
    demo_cartpole_ppo(
        actor=actor,
        critic=critic,
        num_time_steps=18000,
        enable_rendering=True
    )


def test(checkpoint_path: str):
    # Load the trained model
    actor = Actor(state_dim=4, action_dim=1)
    critic = Critic(state_dim=4)
    checkpoint = Checkpoint(
        checkpoint_path, 
        {
            "episode": 0,
            "actor": actor,
            "critic": critic,
        }
    )
    checkpoint.load()
    demo_cartpole_ppo(
        actor=actor,
        critic=critic,
        num_time_steps=5000,
        enable_rendering=False
    )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python main.py <demo|train> <model_path>")
        sys.exit(1)
    command = sys.argv[1]
    model_path = sys.argv[2]
    if command == "train":
        train(model_path)
    elif command == "demo":
        demo(model_path)
    elif command == "test":
        test(model_path)
    else:
        print("Invalid command. Use 'train' or 'demo'.")

