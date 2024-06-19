import torch
import torch.nn as nn


class ArgsSequential(nn.Sequential):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def forward(self, x, *args, **kwargs):
    for mod in self:
      x = mod(x, *args, **kwargs)

    return x

