import numpy as np
import py_misc_utils.alog as alog
import py_misc_utils.assert_checks as tas
import py_misc_utils.np_utils as pyn

from .. import core_utils as cu


class DiscreteAction:

  # NOTE: The voff,von are tuned for nn.Tanh(), make sure this is what the PI net
  # is using as final layer activation.
  def __init__(self, n, start=0, voff=-1.0, von=1.0):
    self._n = n
    self._start = start
    self._voff = voff
    self._von = von
    self._dv = von - voff

  def size(self):
    return self._n

  def rand(self, rand_sigma, action=None):
    if action is None:
      action = np.full(self._n, (self._voff + self._von) / 2, dtype=np.float32)

    rv = action + np.random.normal(scale=rand_sigma, size=self._n) * self._dv / 2

    r = pyn.categorical(rv)

    rvalue = [self._voff] * self._n
    rvalue[r] = self._von

    return rvalue

  def value(self, v):
    return pyn.categorical(v).item() + self._start


class ContinuousAction:

  # NOTE: The vmin,vmax are tuned for nn.Tanh(), make sure this is what the PI net
  # is using as final layer activation.
  def __init__(self, amin, amax, vmin=-1.0, vmax=1.0):
    tas.check_lt(amin, amax, msg=f'Bad ordering')
    tas.check_lt(vmin, vmax, msg=f'Bad ordering')

    self._amin = amin
    self._amax = amax
    self._vmin = vmin
    self._vmax = vmax
    self._da = amax - amin
    self._dv = vmax - vmin

  def size(self):
    return 1

  def rand(self, rand_sigma, action=None):
    if action is None:
      action = (self._vmin + self._vmax) / 2
    else:
      action = cu.item(action)

    rv = action + np.random.normal(scale=rand_sigma) * self._dv / 2

    return [np.clip(rv, self._vmin, self._vmax)]

  def value(self, v):
    v = np.clip(cu.item(v), self._vmin, self._vmax)

    return self._amin + (v - self._vmin) * self._da / self._dv


class Actions:

  def __init__(self, actions):
    self._actions = tuple(actions)

  def __len__(self):
    return len(self._actions)

  def size(self):
    return sum(act.size() for act in self._actions)

  def rand(self, rand_sigma, action=None):
    rvalues = []
    if action is not None:
      spos = 0
      for act in self._actions:
        rvalues.extend(act.rand(rand_sigma, action=action[spos: spos + act.size()]))
        spos += act.size()
    else:
      for act in self._actions:
        rvalues.extend(act.rand(rand_sigma))

    return np.array(rvalues, dtype=np.float32)

  def values(self, action):
    spos, values = 0, []
    for act in self._actions:
      asize = act.size()
      values.append(act.value(action[spos: spos + asize]))
      spos += asize

    return values

