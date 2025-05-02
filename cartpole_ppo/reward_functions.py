"""
This module contains a set of reward functions for reinforcement learning environments.
"""
import numpy as np

def reward_inverted_pendulum(
    qpos: np.ndarray,
    qvel: np.ndarray,
    *,
    alpha_theta: float = 1.0,
    alpha_theta_dot: float = 0.05,
    alpha_x: float = 0.2,
    alpha_x_dot: float = 0.05,
) -> float:
    """
    Calculate the reward based on the vertical position of the pole,
    and it's angular velocity.
    The reward is calculated as:
        reward = alpha_theta * cos(theta) - alpha_theta_dot * theta_dot^2
            - alpha_x * x^2 - alpha_x_dot * x_dot^2
    where theta is the angle of the pole, theta_dot is the angular velocity,
    x is the horizontal position of the cart, and x_dot is the horizontal velocity.

    Args:
        qpos (np.ndarray): The position of the cart and pole.
            [x, theta]
        qvel (np.ndarray): The velocity of the cart and pole.
            [x_dot, theta_dot]
        alpha_theta (float): Weight for the angle of the pole.
        alpha_theta_dot (float): Weight for the angular velocity of the pole.
        alpha_x (float): Weight for the horizontal position of the cart.
        alpha_x_dot (float): Weight for the horizontal velocity of the cart.
    Returns:
        float: The calculated reward.
    """
    x = qpos[0]
    theta = qpos[1]
    x_dot = qvel[0]
    theta_dot = qvel[1]

    reward = (
        alpha_theta * np.cos(theta)
        - alpha_theta_dot * (theta_dot ** 2)
        - alpha_x * (x ** 2)
        - alpha_x_dot * (x_dot ** 2)
    )

    return reward
