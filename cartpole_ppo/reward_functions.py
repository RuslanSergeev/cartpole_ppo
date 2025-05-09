"""
This module contains a set of reward functions the CartPole environment.
User may try to use these reward functions to train the agent.
"""
import numpy as np

def reward_inverted_pendulum(
    qpos: np.ndarray,
    qvel: np.ndarray,
    *,
    alpha_theta: float = 1.0,
    alpha_theta_dot: float = 1e-5,
    alpha_x: float = 1e-2,
    alpha_x_dot: float = 1e-5,
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

def reward_target_point(
    qpos: np.ndarray,
    qvel: np.ndarray,
    *,
    target_pos: np.ndarray,
    alpha_theta: float = 1.0,
    alpha_pos: float = 1.0,
):
    pass
