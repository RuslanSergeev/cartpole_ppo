import time
from typing import Optional, Callable, Any
import mujoco
import mujoco.viewer
import numpy as np
from torch import nn

from .state_generators import get_pendulum_down_state
from .reward_functions import reward_inverted_pendulum
from .ppo_agent import test_agent


class InvertedPendulumEnv:
    def __init__(
        self,
        model_path="mujoco_environments/inverted_pendulum.xml",
        *,
        initial_state_generator:Callable[..., np.ndarray] = get_pendulum_down_state,
        reward_generator:Callable[..., float] = reward_inverted_pendulum,
        enable_rendering: bool = True,
        delta_time: float = 0.01,
    ):
        self.initial_state_generator = initial_state_generator
        self.reward_generator = reward_generator
        self.enable_rendering = enable_rendering
        self.model_path = model_path
        self.init_mujoco()
        self.target_position_updated_time = 0.0
        self.reset()
        self.set_dt(delta_time)

    def __del__(self):
        if self.enable_rendering:
            self.viewer.close()

    def init_mujoco(self):
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        if self.enable_rendering:
            self.viewer = mujoco.viewer.launch_passive(
                self.model, 
                self.data
            )

    def step(self, a):
        self.data.ctrl = a
        mujoco.mj_step(self.model, self.data)
        self.sync()
        reward = self.reward_generator(self.data.qpos, self.data.qvel)
        ob = self.obs()
        terminated = bool(not np.isfinite(ob).all())
        # If rendering, ensure real-time simulation
        if self.enable_rendering:
            time.sleep(self.model.opt.timestep)
        return ob, reward, terminated

    def sync(self):
        if self.enable_rendering:
            self.viewer.sync()

    def obs(self):
        qpos = self.data.qpos
        qvel = self.data.qvel
        # Limit the theta observation to the range [-pi, pi]
        qpos[1] = np.arctan2(np.sin(qpos[1]), np.cos(qpos[1]))
        return np.concatenate([qpos, qvel]).ravel()

    def reset(self, state: Optional[np.ndarray] = None):
        if state is None:
            state_init = self.initial_state_generator()
            self.data.qpos = state_init.ravel()[:2]
            self.data.qvel = state_init.ravel()[2:]
        else:
            self.data.qpos = state.ravel()[:2]
            self.data.qvel = state.ravel()[2:]
        return self.obs()

    def set_dt(self, new_dt):
        """Sets simulations step"""
        self.model.opt.timestep = new_dt

    def set_target_position(self, target_position: np.ndarray):
        self.target_position = target_position
        self.target_position_updated_time = self.current_time
        self.draw_ball()

    def draw_ball(
        self, 
        *,
        color=[1, 0, 0, 1], 
        radius=0.05
    ):
        if self.enable_rendering:
            mujoco.mjv_initGeom(
                self.viewer.user_scn.geoms[0],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[radius, 0, 0],
                pos=np.array(self.target_position),
                mat=np.eye(3).flatten(),
                rgba=np.array(color),
            )
            self.viewer.user_scn.ngeom = 1

    @property
    def current_time(self):
        return self.data.time


def demo_cartpole_ppo(
    actor: nn.Module,
    critic: nn.Module,
    num_time_steps: int = 5000, 
    enable_rendering: bool = True,
):
    """
    Run a demo of the PPO agent on the CartPole environment.
    """
    # Load the actor and critic
    environment = InvertedPendulumEnv(
        enable_rendering=enable_rendering,
        initial_state_generator=get_pendulum_down_state,
    )
    environment.reset()
    # Test the agent
    test_agent(
        actor=actor,
        critic=critic,
        env=environment,
        num_time_steps=num_time_steps,
    )



def main(enable_rendering: bool = True):
    env = InvertedPendulumEnv(enable_rendering=enable_rendering)
    env.set_dt(0.01)
    env.reset()
    try:
        while env.current_time < 100:
            if env.current_time > env.target_position_updated_time + 5:
                target_pos = [np.random.rand() - 0.5, 0, 0.6]
                env.set_target_position(target_pos)
            ob, reward, terminated = env.step(0)
            print(f"time: {env.current_time}, reward: {reward}, ob: {ob}, terminated: {terminated}")
    except:
        env.sync()


if __name__ == "__main__":
    main(enable_rendering=True)
