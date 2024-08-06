import numpy as np
import py_misc_utils.alog as alog
import torch
import torch.optim as optim


class ReduceOnPlateau:

  def __init__(self, optimizer, num_samples, patience_span=None, **kwargs):
    patience = kwargs.get('patience', 10)
    patience_span = patience_span or 0.05

    self.sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    self.sched_samples = int(num_samples * patience_span / patience)
    self.samples = 0
    self.losses = []

  def state_dict(self):
    state = self.__dict__.copy()
    state['sched'] = self.sched.state_dict()

    return state

  def load_state_dict(self, state):
    self.sched.load_state_dict(state.pop('sched'))
    self.__dict__.update(state)

  def step(self, batch_samples, batch_loss):
    self.samples += batch_samples
    self.losses.append(batch_loss)
    if self.samples >= self.sched_samples:
      self.sched.step(np.mean(self.losses))
      self.samples = 0
      self.losses = []

