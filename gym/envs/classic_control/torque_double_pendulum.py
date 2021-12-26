from typing import Optional
from os import path

import torch
import numpy as np
from gym import core, spaces

class TorqueDoublePendulum(core.Env):
    g = 9.81

    def __init__(self):
        self.viewer = None

        self.max_speed = 8
        self.max_torque = 2.0

        # self.torques = np.array([-self.max_torque, 0, self.max_torque])

        # Mass of each rod
        self.m = 1.0
        # Length of each rod
        self.l = 1.0
        # Time discretization
        self.dt = 0.05

        high = np.array([np.pi, self.max_speed, np.pi, self.max_speed])
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float64)

        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float64
        )
        # self.action_space = spaces.Discrete(3)

    def reset(self, seed: Optional[int] = None):
        # super().reset(seed=seed)
        high = np.array([np.pi/8, 1, np.pi/8, 1])
        state = np.random.uniform(low=-high, high=high)
        self.state = torch.tensor(state)
        # self.state = state
        # self.last_u = None
        return self.state

    def step(self, action):
        theta_1, theta_1_dot, theta_2, theta_2_dot = self.state

        # torque = self.torques[action]
        torque = torch.tensor(action).clamp(-self.max_torque, self.max_torque)
        # torque = np.clip(torque, -self.max_torque, self.max_torque)


        cost = angle_normalize(theta_2) ** 2 + 2*angle_normalize(theta_1) ** 2 # + 0.01 * (theta_2_dot ** 2 + theta_1_dot ** 2) + 0.0001 * (torque ** 2)

        theta_1_dot_dot = (-4 * theta_2_dot ** 2 * torch.sin(theta_1 - theta_2) - (2 * theta_1_dot ** 2 * torch.sin(theta_1 - theta_2) + 19.64 * torch.sin(theta_2))*torch.cos(theta_1 - theta_2) + 19.64 * torch.sin(theta_1))/(4 - 2*torch.cos(theta_1 - theta_2)**2)
        theta_2_dot_dot = (2 * theta_1_dot ** 2 * torch.sin(theta_1 - theta_2) + torch.sin(2*theta_1 - 2*theta_2) * theta_2_dot ** 2 + 14.73 * torch.sin(theta_2) - 4.91 * torch.sin(2*theta_1 - theta_2))/(4 - 2*torch.cos(theta_1 - theta_2)**2) + torque
        # theta_1_dot_dot = (-4 * theta_2_dot ** 2 * np.sin(theta_1 - theta_2) - (2 * theta_1_dot ** 2 * np.sin(theta_1 - theta_2) + 19.64 * np.sin(theta_2))*np.cos(theta_1 - theta_2) + 19.64 * np.sin(theta_1))/(4 - 2*np.cos(theta_1 - theta_2)**2)
        # theta_2_dot_dot = (2 * theta_1_dot ** 2 * np.sin(theta_1 - theta_2) + np.sin(2*theta_1 - 2*theta_2) * theta_2_dot ** 2 + 14.73 * np.sin(theta_2) - 4.91 * np.sin(2*theta_1 - theta_2))/(4 - 2*np.cos(theta_1 - theta_2)**2) + torque


        # TODO: Add some noise?
        new_theta_1_dot = theta_1_dot + theta_1_dot_dot*self.dt
        new_theta_1_dot = new_theta_1_dot.clamp(-self.max_speed, self.max_speed)
        # new_theta_1_dot = np.clip(new_theta_1_dot, -self.max_speed, self.max_speed)

        # TODO: Add some noise?
        new_theta_2_dot = theta_2_dot + theta_2_dot_dot*self.dt
        new_theta_2_dot = new_theta_2_dot.clamp(-self.max_speed, self.max_speed)
        # new_theta_2_dot = np.clip(new_theta_2_dot, -self.max_speed, self.max_speed)

        # Friction
        new_theta_1_dot *= 0.98
        new_theta_2_dot *= 0.98

        new_theta_1 = theta_1 + theta_1_dot * self.dt
        new_theta_2 = theta_2 + theta_2_dot * self.dt

        self.state = torch.tensor([new_theta_1, new_theta_1_dot, new_theta_2, new_theta_2_dot])
        # self.state = np.array([new_theta_1, new_theta_1_dot, new_theta_2, new_theta_2_dot])

        # done = max(abs(new_theta_1), abs(new_theta_2)) > np.pi/6 
        return self.state, -cost.item(), False, {}

    def render(self, mode="human"):
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)

            inner_rod = rendering.make_capsule(1, 0.2)
            inner_rod.set_color(0.8, 0.3, 0.3)
            self.inner_pole_transform = rendering.Transform()
            inner_rod.add_attr(self.inner_pole_transform)
            self.viewer.add_geom(inner_rod)

            outer_rod = rendering.make_capsule(1, 0.2)
            outer_rod.set_color(0.3, 0.3, 0.8)
            self.outer_pole_transform = rendering.Transform()
            outer_rod.add_attr(self.outer_pole_transform)
            self.viewer.add_geom(outer_rod)

            axle = rendering.make_circle(0.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)

            # fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            # self.img = rendering.Image(fname, 1.0, 1.0)
            # self.imgtrans = rendering.Transform()
            # self.img.add_attr(self.imgtrans)

        # self.viewer.add_onetime(self.img)
        angle = self.state[2] + np.pi / 2
        self.inner_pole_transform.set_rotation(angle)

        self.outer_pole_transform.set_translation(np.cos(angle), np.sin(angle))
        self.outer_pole_transform.set_rotation(self.state[0] + np.pi / 2)

        # if self.last_u is not None:
        #     self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi