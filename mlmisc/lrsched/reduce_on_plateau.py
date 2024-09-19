import numpy as np
import py_misc_utils.alog as alog
import py_misc_utils.utils as pyu
import torch
import torch.optim as optim


class ReduceOnPlateau:

  STATE_FIELDS = ()

  def __init__(self, optimizer, num_batches, patience_span=None, **kwargs):
    patience = kwargs.get('patience', 10)
    patience_span = patience_span or 0.1

    self.sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    self.sched_batches = int(num_batches * patience_span / patience)
    self.batchno = 0
    self.losses = []

  def state_dict(self):
    state = {k: getattr(self, k) for k in self.STATE_FIELDS}
    state['sched'] = self.sched.state_dict()

    return state

  def load_state_dict(self, state):
    self.sched.load_state_dict(state.pop('sched'))
    self.__dict__.update(state)

  def train_step(self, batch_loss):
    self.losses.append(batch_loss)
    self.batchno += 1
    if self.batchno >= self.sched_batches:
      self.sched.step(np.mean(self.losses))
      self.batchno = 0
      self.losses = []
      alog.debug(f'Last LR is {pyu.format(self.sched.get_last_lr(), ".3e")}')

  def epoch_step(self, val_loss):
    loss = float('nan') if val_loss is None else val_loss
    alog.debug(f'Scheduler step called with {loss:.4f} validation loss')

  def step(self):
    pass

