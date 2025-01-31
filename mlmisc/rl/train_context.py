import contextlib
import datetime
import time
import types

import numpy as np
import py_misc_utils.alog as alog
import py_misc_utils.assert_checks as tas
import py_misc_utils.moving_average as pyma


class TrainContext:

  def __init__(self, target_reward,
               action_noise_base=0.5,
               action_noise_end=0.05,
               warmup_steps=100,
               ma_factor=0.95):
    self._target_reward = target_reward
    self._action_noise_base = action_noise_base
    self._action_noise_end = action_noise_end
    self._base_reward = None
    self._warmup_steps = warmup_steps
    self._sample_time = self._train_time = 0
    self.stepno = 0
    self.train_stepno = 0
    self._reward_ma = pyma.MovingAverage(ma_factor, init=0)
    self._episode_steps_ma = pyma.MovingAverage(ma_factor, init=0)

  @contextlib.contextmanager
  def sampling(self):
    optime = types.SimpleNamespace(start=time.time(), elapsed=None)
    try:
      yield optime
    finally:
      optime.elapsed = time.time() - optime.start
      self._sample_time += optime.elapsed

  @contextlib.contextmanager
  def training(self):
    optime = types.SimpleNamespace(start=time.time(), elapsed=None)
    try:
      yield optime
    finally:
      optime.elapsed = time.time() - optime.start
      self._train_time += optime.elapsed

  def sampling_time(self):
    return datetime.timedelta(seconds=self._sample_time)

  def training_time(self):
    return datetime.timedelta(seconds=self._train_time)

  def reward_ma(self):
    return self._reward_ma.value

  def episode_steps_ma(self):
    return self._episode_steps_ma.value

  def reset(self,
            target_reward=None,
            action_noise_base=None,
            action_noise_end=None):
    if target_reward is not None:
      self._target_reward = target_reward
    if action_noise_base is not None:
      self._action_noise_base = action_noise_base
    if action_noise_end is not None:
      self._action_noise_end = action_noise_end

  def action_noise(self):
    if self._base_reward is None:
      if self._warmup_steps > self.stepno:
        return self._action_noise_base

      self._base_reward = self.reward_ma()

    noise_x = (self._base_reward, self._target_reward)
    noise_y = (self._action_noise_base, self._action_noise_end)

    return np.interp(self.reward_ma(), noise_x, noise_y)

  def step(self, reward, episode_steps):
    self._reward_ma.update(reward)
    self._episode_steps_ma.update(episode_steps)
    self.stepno += 1

  def train_step(self):
    self.train_stepno += 1

