import functools

import py_misc_utils.alog as alog
import py_misc_utils.assert_checks as tas
import py_misc_utils.fin_wrap as pyfw
import py_misc_utils.fs_utils as pyfsu
import torch
import torch.nn as nn

from .. import core_utils as cu
from .. import utils as ut


class NoopParamUpdater:

  def update(self, stepno):
    pass


class ParamUpdater(NoopParamUpdater):

  def __init__(self, source_net, target_net):
    super().__init__()
    self._source_net = source_net
    self._target_net = target_net


# Even though this is advertised by many RL writeups, in my experience it is a huge
# waste of time. Moving to the CheckpointParamUpdater suddenly made all my tests
# converge, whereas before they almost never did, or if they did, it was very
# inconsistent. And if you think about it, it makes sense that if A and B are two
# good states, somewhere linearly in the middle might not be.
class LinearParamUpdater(ParamUpdater):

  def __init__(self, source_net, target_net, steps_per_update, tau):
    super().__init__(source_net, target_net)
    self._steps_per_update = steps_per_update
    self._tau = tau
    self._next_update_step = None

  def update(self, stepno):
    if self._next_update_step is None:
      self._next_update_step = stepno + self._steps_per_update
    elif stepno >= self._next_update_step:
      alog.info(f'[{stepno}] Updating ({self._steps_per_update} new train steps) ...')
      cu.update_params(self._source_net, self._target_net, tau=self._tau)
      self._next_update_step = stepno + self._steps_per_update


class CheckpointParamUpdater(ParamUpdater):

  def __init__(self, source_net, target_net, steps_per_update, device=None):
    super().__init__(source_net, target_net)
    self._steps_per_update = steps_per_update
    self._device = device
    self._next_update_step = None

    tmp_path = pyfsu.temp_path()
    pyfw.fin_wrap(self, '_tmp_path', tmp_path,
                  finfn=functools.partial(pyfsu.maybe_remove, tmp_path))

  def update(self, stepno):
    if self._next_update_step is None:
      self._next_update_step = stepno + self._steps_per_update
      ut.model_save(self._source_net, self._tmp_path)
    elif stepno >= self._next_update_step:
      alog.info(f'[{stepno}] Updating ({self._steps_per_update} new train steps) ...')
      ut.model_load(self._tmp_path, model=self._target_net, device=self._device)
      ut.model_save(self._source_net, self._tmp_path)
      self._next_update_step = stepno + self._steps_per_update

