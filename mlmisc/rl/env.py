import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import py_misc_utils.alog as alog
import py_misc_utils.assert_checks as tas
import py_misc_utils.fin_wrap as pyfw

from .. import core_utils as cu

from . import actions as rlact


def _build_actions(env):
  actions = []
  aspace = env.action_space
  if isinstance(aspace, gym.spaces.Box):
    tas.check_eq(len(aspace.shape), 1, msg=f'Shape mismatch: {aspace}')
    for i in range(aspace.shape[0]):
      amin, amax = cu.item(aspace.low[i]), cu.item(aspace.high[i])
      actions.append(rlact.ContinuousAction(amin, amax))
  elif isinstance(aspace, gym.spaces.Discrete):
    actions.append(rlact.DiscreteAction(cu.item(aspace.n), start=cu.item(aspace.start)))
  else:
    alog.xraise(RuntimeError, f'Unknown action space: {aspace}')

  return rlact.Actions(actions)


ALIVE = 0.0
DONE = 1.0
TERMINATED = -1.0

class EnvBase:

  def __init__(self, env):
    pyfw.fin_wrap(self, '_env', env, finfn=env.close)
    self._actions = _build_actions(env)

  def close(self):
    pyfw.fin_wrap(self, '_env', None, cleanup=True)

  def get_screen(self):
    screen = self._env.render().transpose((2, 0, 1))

    return np.ascontiguousarray(screen, dtype=np.float32) / 255

  def print_screen(self, screen=None, figsize=(5, 5), title=None, path=None):
    if screen is None:
      screen = self.get_screen()
    plt.figure(figsize=figsize)
    plt.imshow(screen.transpose((1, 2, 0)), interpolation='bilinear')
    if title:
      plt.title(title)
    if path is not None:
      plt.savefig(path)
    else:
      plt.show()

  def reset(self, **kwargs):
    state, info = self._env.reset(**kwargs)

    return np.asarray(state, dtype=np.float32)

  def step(self, action):
    actions = self.action_values(action)
    env_action = actions[0] if len(actions) == 1 else actions

    next_state, reward, done, terminated, info = self._env.step(env_action)

    return (np.asarray(next_state, dtype=np.float32),
            float(reward),
            DONE if done else TERMINATED if terminated else ALIVE)

  def rand(self, rand_sigma, action=None):
    return self._actions.rand(rand_sigma, action=action)

  def action_values(self, action):
    return self._actions.values(action)

  def num_actions(self):
    return self._actions.size()

  def num_signals(self):
    return self._env.observation_space.shape[0]


# See https://github.com/clvrai/awesome-rl-envs for usable Gym-based environments.
class GymEnv(EnvBase):

  def __init__(self, name, **kwargs):
    kwargs.update(render_mode='rgb_array')
    env = gym.make(name, **kwargs)

    super().__init__(env)
    self.name = name
    self._kwargs = kwargs

  def new(self):
    return self.__class__(self.name, **self._kwargs)

  def info(self):
    info = []
    info.append(f'Env Args: {self._kwargs}')
    info.append(f'Observation Space: {self._env.observation_space}')
    info.append(f'Action Space: {self._env.action_space}')
    if isinstance(self._env.action_space, gym.spaces.Box):
      for i in range(self._env.action_space.shape[0]):
        info.append(f'  low={self._env.action_space.low[i]:.2e}\t' \
                    f'hi={self._env.action_space.high[i]:.2e}')
    elif isinstance(self._env.action_space, gym.spaces.Discrete):
      info.append(f'  num_actions = {self._env.action_space.n} ' \
                  f'(start={self._env.action_space.start})')

    return '\n'.join(info)

