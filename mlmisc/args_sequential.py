import torch
import torch.nn as nn

from . import args_base as ab


class ArgsSequential(ab.ArgsBase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def forward(self, x, *args, **kwargs):
    for net in self.values():
      x = net(x, *args, **kwargs)

    return x

