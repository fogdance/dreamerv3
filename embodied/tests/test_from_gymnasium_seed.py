import gymnasium as gym
import numpy as np
from gymnasium import spaces

from embodied.envs.from_gymnasium import FromGymnasium


class SeedRecordingEnv(gym.Env):

  def __init__(self):
    self.observation_space = spaces.Box(
        low=0, high=255, shape=(1,), dtype=np.uint8)
    self.action_space = spaces.Discrete(2)
    self.reset_seeds = []

  def reset(self, seed=None, options=None):
    del options
    self.reset_seeds.append(seed)
    return np.asarray([0], np.uint8), {}

  def step(self, action):
    del action
    return np.asarray([0], np.uint8), 0.0, True, False, {}


def test_reset_seed_is_only_used_for_first_reset():
  base = SeedRecordingEnv()
  env = FromGymnasium(base, reset_seed=123)

  env.step({'reset': np.asarray(True)})
  env.step({'reset': np.asarray(True)})

  assert base.reset_seeds == [123, None]
