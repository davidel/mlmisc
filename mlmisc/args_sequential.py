import torch
import torch.nn as nn


class ArgsSequential(nn.Sequential):

  def forward(self, x, *args, **kwargs):
    for mod in self:
      x = mod(x, *args, **kwargs)

    return x

