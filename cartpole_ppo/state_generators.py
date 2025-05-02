import numpy as np

def get_random_state(
    x_init_min: float = -1.0,
    x_init_max: float = 1.0,
    theta_init_min: float = -np.pi,
    theta_init_max: float = np.pi,
    x_dot_init_min: float = -1.0,
    x_dot_init_max: float = 1.0,
    theta_dot_init_min: float = -1.0,
    theta_dot_init_max: float = 1.0,
) -> np.ndarray:
    """Utility function to get a random state for the cartpole environment."""
    x_init = np.random.uniform(x_init_min, x_init_max)
    theta_init = np.random.uniform(theta_init_min, theta_init_max)
    x_dot_init = np.random.uniform(x_dot_init_min, x_dot_init_max)
    theta_dot_init = np.random.uniform(
        theta_dot_init_min,
        theta_dot_init_max,
    )
    return np.array([
        x_init,
        theta_init,
        x_dot_init,
        theta_dot_init,
    ])


def get_pendulum_down_state(
    delta_theta: float = 1.0e-3,
) -> np.ndarray:
    """Utility function to get a state for the cartpole environment."""
    # Pendulum down state either
    #   - theta = pi
    #   - theta = -pi
    delta = np.random.uniform(-delta_theta, delta_theta)
    theta = np.random.choice([np.pi+delta, -np.pi+delta])
    return np.array([
        0.0,
        theta,
        0.0,
        0.0,
    ])
