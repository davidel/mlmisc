import inspect

import torch
import torch.nn as nn


class Lambda(nn.Module):

  def __init__(self, fn):
    super().__init__()
    self.fn = fn

  def forward(self, *args, **kwargs):
    return self.fn(*args, **kwargs)

  def extra_repr(self):
    source = inspect.getsource(self.fn)

    return f'source={source}'

