import sys
from cartpole_ppo.ppo_agent import (
    train_cartpole_ppo,
    test_agent,
    demo_cartpole_ppo,
)
from cartpole_ppo.hardware_manager import Hardware_manager


def train(checkpoint_path: str):
    train_cartpole_ppo(
        model_checkpoint_path=checkpoint_path,
        num_episodes=5000,
        num_actors=5,
        num_epochs=10,
        num_time_steps=6000,
        log_any=100,
        device=Hardware_manager.get_device()
    )


def demo(checkpoint_path: str):
    demo_cartpole_ppo(
        checkpoint_path=checkpoint_path, 
        num_time_steps=18000,
        enable_rendering=True
    )


def test(checkpoint_path: str):
    demo_cartpole_ppo(
        checkpoint_path=checkpoint_path, 
        num_time_steps=18000,
        enable_rendering=False
    )
    return True


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

