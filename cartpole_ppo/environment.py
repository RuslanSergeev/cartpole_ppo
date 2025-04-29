import time
from typing import Optional
import mujoco
import mujoco.viewer
import numpy as np

from .reward_functions import reward_inverted_pendulum


class InvertedPendulumEnv:
    def __init__(
        self,
        model_path="mojoco_environments/inverted_pendulum.xml",
        *,
        enable_rendering: bool = True,
        delta_time: float = 0.01,
    ):
        self.enable_rendering = enable_rendering
        self.model_path = model_path
        self.init_mujoco()
        self.target_position_updated_time = 0.0
        self.init_qpos = np.zeros(2)
        self.init_qvel = np.zeros(2)
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
        self.data.ctrl = 3*a
        mujoco.mj_step(self.model, self.data)
        self.sync()
        reward = reward_inverted_pendulum(
            self.data.qpos,
            self.data.qvel,
            alpha_theta=1.0,
            alpha_theta_dot=0,
            alpha_x=2,
            alpha_x_dot=0,
        )
        ob = self.obs()
        terminated = bool(not np.isfinite(ob).all())
        if self.enable_rendering:
            time.sleep(self.model.opt.timestep)
        return ob, reward, terminated

    def sync(self):
        if self.enable_rendering:
            self.viewer.sync()

    def obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).ravel()

    def reset(self, state: Optional[np.ndarray] = None):
        if state is None:
            self.data.qpos = self.init_qpos
            self.data.qvel = self.init_qvel
        else:
            self.data.qpos = state[:2]
            self.data.qvel = state[2:]
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


def set_random_target_position(env):
    if env.current_time - env.target_position_updated_time> 5:
        target_pos = [np.random.rand() - 0.5, 0, 0.6]
        env.set_target_position(target_pos)


def main(enable_rendering: bool = True):
    env = InvertedPendulumEnv(enable_rendering=enable_rendering)
    env.set_dt(0.01)
    env.reset()
    try:
        while env.current_time < 100:
            set_random_target_position(env)
            ob, reward, terminated = env.step(0)
            print(f"time: {env.current_time}, reward: {reward}, ob: {ob}, terminated: {terminated}")
    except:
        env.sync()


if __name__ == "__main__":
    main(enable_rendering=True)
