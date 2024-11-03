import torch
import torch.nn as nn

from . import nets_dict as netd


class ArgsSequential(netd.NetsDict):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def forward(self, x, *args, **kwargs):
    for net in self.values():
      x = net(x, *args, **kwargs)

    return x

