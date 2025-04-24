import math
from typing import Optional, Union

import numpy as np

import gym
from gym import logger, spaces
from gym.envs.classic_control import utils
from gym.error import DependencyNotInstalled

class SimpleAdaptedMuscle:
    """
    Equations:
    -- Muscle synapse
    dCn(t)/dt + Cn(t)/tau_c = u(t), u - input
    x(t) = Cn^m/(Cn^m + k^m)
    -- Output Force
    dF(t)/dt + F(t)/tau_1 = Ax(t)
    """

    PARAMETER_SETS = {
        'set1': {'tau_c': 1/71, 'tau_1': 1/130, 'm': 2.5, 'k': 0.75, 'A': 0.0074},
        'set2': {'tau_c': 1/60, 'tau_1': 1/120, 'm': 3.0, 'k': 0.80, 'A': 0.0080},
        'set3': {'tau_c': 1/80, 'tau_1': 1/140, 'm': 2.0, 'k': 0.70, 'A': 0.0065},
    }

    def __init__(self, parameter_set, **kwargs):
        """
        Arguments:
        w - weight of neuron-muscle synapse
        parameter_set - set of parameters to use
        """
        params = self.PARAMETER_SETS[parameter_set]
        self.tau_c = params['tau_c']
        self.tau_1 = params['tau_1']
        self.m = params['m']
        self.k = params['k']
        self.A = params['A']

        self.w = kwargs.get('w', 0.5)
        self.A = self.A * kwargs.get('N', 10)
        self.Cn = 0
        self.Cn_prev = 0
        self.F = 0
        self.F_prev = 0
        self.x = 0

    def set_init_conditions(self):
        self.Cn = 0
        self.Cn_prev = 0
        self.F = 0
        self.F_prev = 0
        self.x = 0

    def step(self, dt=0.1, u=0):
        self.Cn = self.Cn_prev + dt * (self.w * u - self.Cn_prev * self.tau_c)
        self.x = self.Cn ** self.m / (self.Cn ** self.m + self.k ** self.m)
        self.F = self.F_prev + dt * (self.A * self.x - self.F_prev * self.tau_1)
        self.F_prev = self.F
        self.Cn_prev = self.Cn
        return self.F

class MusclePoleEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    """
    ### Description

    This environment corresponds to a pole that is controlled directly by muscles.
    The goal is to balance the pole by applying forces using muscle models.

    ### Action Space

    The action is a `ndarray` with shape `(1,)` which can take values `{0, 1}` indicating the direction
    of the force applied by the muscles.

    | Num | Action                 |
    |-----|------------------------|
    | 0   | Apply force to the left|
    | 1   | Apply force to the right|

    ### Observation Space

    The observation is a `ndarray` with shape `(2,)` with the values corresponding to the following positions and velocities:

    | Num | Observation           | Min                 | Max               |
    |-----|-----------------------|---------------------|-------------------|
    | 0   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
    | 1   | Pole Angular Velocity | -Inf                | Inf               |

    ### Rewards

    Since the goal is to keep the pole upright for as long as possible, a reward of `+1` for every step taken,
    including the termination step, is allotted.

    ### Starting State

    All observations are assigned a uniformly random value in `(-0.05, 0.05)`

    ### Episode End

    The episode ends if any one of the following occurs:

    1. Termination: Pole Angle is greater than ±12°
    2. Truncation: Episode length is greater than 500

    ### Arguments

    ```
    gym.make('MusclePole-v1')
    ```

    No additional arguments are currently supported.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, render_mode: Optional[str] = None, muscle_parameter_set: str = 'set1'):
        self.gravity = 9.8
        self.masspole = 0.1
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360

        high = np.array(
            [
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None

        self.steps_beyond_terminated = None

        # Initialize muscle models for left and right forces
        self.muscle_left = SimpleAdaptedMuscle(parameter_set=muscle_parameter_set)
        self.muscle_right = SimpleAdaptedMuscle(parameter_set=muscle_parameter_set)

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."
        theta, theta_dot = self.state

        # Determine the force based on the muscle model
        if action == 0:
            force = -self.muscle_left.step(dt=self.tau)
        else:
            force = self.muscle_right.step(dt=self.tau)

        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # Calculate the angular acceleration
        thetaacc = (self.gravity * sintheta - force * costheta) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.masspole)
        )

        if self.kinematics_integrator == "euler":
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (theta, theta_dot)

        terminated = bool(
            theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not terminated:
            reward = 1.0
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            reward = 1.0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            reward = 0.0

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = utils.maybe_parse_reset_bounds(
            options, -0.05, 0.05  # default low
        )  # default high
        self.state = self.np_random.uniform(low=low, high=high, size=(2,))
        self.steps_beyond_terminated = None

        # Reset muscle models
        self.muscle_left.set_init_conditions()
        self.muscle_right.set_init_conditions()

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = 1  # The pole is centered
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)

        if self.state is None:
            return None

        theta = self.state[0]

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-theta)
            coord = (coord[0] + self.screen_width / 2, coord[1] + self.screen_height / 2)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(self.screen_width / 2),
            int(self.screen_height / 2),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(self.screen_width / 2),
            int(self.screen_height / 2),
            int(polewidth / 2),
            (129, 132, 203),
        )

        # Преобразуем self.screen_height / 2 в целое число
        gfxdraw.hline(self.surf, 0, self.screen_width, int(self.screen_height / 2), (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False
