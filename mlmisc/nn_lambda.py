import functools

import torch
import torch.nn as nn


class Lambda(nn.Module):

  def __init__(self, fn, *args, _info=None, **kwargs):
    super().__init__()
    self.fn = functools.partial(fn, *args, **kwargs)
    self.info = _info

  def forward(self, *args, **kwargs):
    return self.fn(*args, **kwargs)

  def extra_repr(self):
    return self.info or repr(self.fn)

