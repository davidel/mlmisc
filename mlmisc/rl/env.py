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


class EnvBase:

  def __init__(self, name, env):
    pyfw.fin_wrap(self, '_env', env, finfn=env.close)
    self.name = name
    self._actions = _build_actions(env)

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

  def reset(self):
    state, info = self._env.reset()

    return np.asarray(state, dtype=np.float32)

  def step(self, action):
    actions = self.action_values(action)
    env_action = actions[0] if len(actions) == 1 else actions

    next_state, reward, done, terminated, info = self._env.step(env_action)

    return (np.asarray(next_state, dtype=np.float32),
            float(reward),
            1.0 if done else -1.0 if terminated else 0.0)

  def rand(self, rand_sigma, action=None):
    return self._actions.rand(rand_sigma, action=action)

  def action_values(self, action):
    return self._actions.values(action)

  def num_actions(self):
    return self._actions.size()

  def num_signals(self):
    return self._env.observation_space.shape[0]


class GymEnv(EnvBase):

  def __init__(self, name, **kwargs):
    kwargs.update(render_mode='rgb_array')
    env = gym.make(name, **kwargs).unwrapped

    alog.debug(f'Env Args: {kwargs}')
    alog.debug(f'Observation Space: {env.observation_space}')
    alog.debug(f'Action Space: {env.action_space}')
    if isinstance(env.action_space, gym.spaces.Box):
      for i in range(env.action_space.shape[0]):
        alog.debug(f'  low={env.action_space.low[i]:.2e}\thi={env.action_space.high[i]:.2e}')
    elif isinstance(env.action_space, gym.spaces.Discrete):
      alog.debug(f'  num_actions = {env.action_space.n} (start={env.action_space.start})')

    super().__init__(name, env)

