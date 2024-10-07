import torch
import torch.nn as nn


class Lambda(nn.Module):

  def __init__(self, fn, info=None):
    super().__init__()
    if isinstance(fn, str):
      self.fn = eval(f'lambda {fn}')
      self.info = info or f'lambda {fn}'
    else:
      self.fn = fn
      self.info = info

  def forward(self, *args, **kwargs):
    return self.fn(*args, **kwargs)

  def extra_repr(self):
    return self.info or ''

