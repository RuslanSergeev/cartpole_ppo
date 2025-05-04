import sys
from cartpole_ppo.environment import InvertedPendulumEnv as Environment
from cartpole_ppo.actor import Actor
from cartpole_ppo.critic import Critic
from cartpole_ppo.hardware_manager import Hardware_manager
from cartpole_ppo.state_generators import (
    get_pendulum_down_state,
    get_pendulum_random_state,
)
from cartpole_ppo.ppo import PPO_agent


def train(
    checkpoint_path: str,
    device = Hardware_manager.get_device(),
):
    # Prepare the environment
    environment = Environment(enable_rendering=False)
    # Actor network for policy approximation
    actor = Actor(state_dim=4, action_dim=1).to(device)
    # Critic network for value approximation
    critic = Critic(state_dim=4).to(device)
    # PPO agent
    ppo_agent = PPO_agent(
        environment, actor, critic,
        gamma=0.99, lam=0.95, epsilon=0.2,
        lr_actor=1e-4, lr_critic=0.5e-4,
        train_init_state_generator=get_pendulum_random_state,
        test_init_state_generator=get_pendulum_down_state,
        device=device
    )
    # Train the PPO agent
    ppo_agent.train(
        num_episodes=1000,
        num_actors=1,
        num_time_steps=6000,
        num_epochs=10,
        batch_size=256,
        log_any=5,
        model_checkpoint_path=checkpoint_path,
    )


def demo(
    checkpoint_path: str,
    device = Hardware_manager.get_device()
):
    actor = Actor(state_dim=4, action_dim=1).to(device)
    critic = Critic(state_dim=4).to(device)
    environment = Environment(enable_rendering=True)
    ppo_agent = PPO_agent(
        environment, actor, critic,
        train_init_state_generator=get_pendulum_random_state,
        test_init_state_generator=get_pendulum_down_state,
    )
    ppo_agent.load(checkpoint_path)
    ppo_agent.test(num_time_steps=12000)


def test(checkpoint_path: str):
    # Load the trained model
    actor = Actor(state_dim=4, action_dim=1)
    critic = Critic(state_dim=4)
    environment = Environment(enable_rendering=False)
    ppo_agent = PPO_agent(
        environment, actor, critic,
        train_init_state_generator=get_pendulum_random_state,
        test_init_state_generator=get_pendulum_down_state,
    )
    ppo_agent.load(checkpoint_path)
    # Test the agent
    ppo_agent.test(num_time_steps=12000)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python main.py <demo|train> <model_path>")
        sys.exit(1)
    command = sys.argv[1]
    model_path = sys.argv[2]
    if command == "--train":
        train(model_path)
    elif command == "--demo":
        demo(model_path)
    elif command == "--test":
        test(model_path)
    else:
        print("Invalid command. Use 'train' or 'demo'.")
