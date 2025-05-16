import mujoco
import mujoco.viewer
import time
from typing import Optional, Callable
import numpy as np

from .state_generators import (
    get_pendulum_down_state, 
    get_pendulum_random_state,
    get_initial_target_position,
    get_random_target_position
)
from .reward_functions import reward_inverted_pendulum


class InvertedPendulumEnv:
    def __init__(
        self,
        model_path="mujoco_environments/inverted_pendulum.xml",
        *,
        reward_generator:Callable[..., float] = reward_inverted_pendulum,
        target_position_generator: Callable[..., np.ndarray] = get_initial_target_position,
        enable_rendering: bool = True,
        delta_time: float = 0.01,
        constant_x_offset: float = 0.19,
        target_position_update_period: float = 5.0,
    ):
        self.reward_generator = reward_generator
        self.target_position_generator = target_position_generator
        self.target_position_update_period = target_position_update_period
        self.constant_x_offset = constant_x_offset
        self.enable_rendering = enable_rendering
        self.model_path = model_path
        self.init_mujoco()
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
        self.set_target_position()
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
        obs_qpos = self.data.qpos.copy()
        obs_qvel = self.data.qvel.copy()
        # Limit the theta observation to the range [-pi, pi]
        obs_qpos[1] = np.arctan2(np.sin(obs_qpos[1]), np.cos(obs_qpos[1]))
        if obs_qpos[1] < 30/180*np.pi and obs_qpos[1] > -30/180*np.pi:
            self.stable_count += 1
            if self.stable_count > 100:
                obs_qpos[0] = (
                    obs_qpos[0] - self.target_position[0] + self.constant_x_offset
                )
        else:
            self.stable_count = 0
        return np.concatenate([obs_qpos, obs_qvel])

    def reset(self, state: Optional[np.ndarray] = None):
        self.data.time = 0.0
        self.stable_count = 0
        self.target_position_updated_time = 0.0
        self.target_position = np.array([0.0, 0.0, 0.6])
        if state is None:
            state_init = get_pendulum_down_state(delta_theta=1e-10)
            self.data.qpos = state_init.ravel()[:2]
            self.data.qvel = state_init.ravel()[2:]
        else:
            self.data.qpos = state.ravel()[:2]
            self.data.qvel = state.ravel()[2:]
        return self.obs()

    def set_dt(self, new_dt):
        """Sets simulations step"""
        self.model.opt.timestep = new_dt

    def set_target_position(
        self, 
        target_position: Optional[np.ndarray] = None
    ):
        if self.current_time > (
            self.target_position_updated_time + self.target_position_update_period
        ):
            if target_position is None:
                target_position = self.target_position_generator()
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


def main(enable_rendering: bool = True):
    env = InvertedPendulumEnv(enable_rendering=enable_rendering)
    try:
        while env.current_time < 100:
            if env.current_time > env.target_position_updated_time + 5:
                target_pos = np.array([np.random.rand() - 0.5, 0, 0.6])
                env.set_target_position(target_pos)
            ob, reward, terminated = env.step(0)
            print(f"time: {env.current_time}, reward: {reward}, ob: {ob}, terminated: {terminated}")
    except:
        env.sync()


if __name__ == "__main__":
    main(enable_rendering=True)
