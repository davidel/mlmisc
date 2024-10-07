import torch
import torch.nn as nn


class Lambda(nn.Module):

  def __init__(self, fn, info=None):
    super().__init__()
    self.fn = fn
    self.info = info

  def forward(self, *args, **kwargs):
    return self.fn(*args, **kwargs)

  def extra_repr(self):
    return self.info or ''

