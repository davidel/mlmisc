import numpy as np
import py_misc_utils.alog as alog
import py_misc_utils.utils as pyu
import torch
import torch.optim as optim

from .. import core_utils as cu


class ReduceOnPlateau:

  STATE_FIELDS = ()

  def __init__(self, optimizer, num_batches, patience_span=None, **kwargs):
    patience = kwargs.get('patience', 10)
    patience_span = patience_span or 0.1

    self._sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    self._sched_batches = int(num_batches * patience_span / patience)
    self._batchno = 0
    self._losses = []

  def state_dict(self):
    state = {k: getattr(self, k) for k in self.STATE_FIELDS}
    state['sched'] = self._sched.state_dict()

    return state

  def load_state_dict(self, state):
    self._sched.load_state_dict(state.pop('sched'))
    self.__dict__.update(state)

  def get_last_lr(self):
    return self._sched.get_last_lr()

  def print_lr(self, *args, **kwargs):
    return self._sched.print_lr(*args, **kwargs)

  def train_step(self, batch_loss):
    self._losses.append(cu.item(batch_loss))
    self._batchno += 1
    if self._batchno >= self._sched_batches:
      self._sched.step(np.mean(self._losses))
      self._batchno = 0
      self._losses = []
      alog.debug(f'Last LR is {pyu.format(self._sched.get_last_lr(), ".3e")}')

  def epoch_step(self, val_loss):
    loss = float('nan') if val_loss is None else cu.item(val_loss)
    alog.debug(f'Scheduler step called with {loss:.4f} validation loss')

  def step(self):
    pass

