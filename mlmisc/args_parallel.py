import torch
import torch.nn as nn


class ArgsParallel(nn.Module):

  def __init__(self, *args, cat_dim=None, **kwargs):
    super().__init__()
    self.cat_dim = cat_dim
    self.nets = nn.ModuleDict()
    net_list = args[0] if len(args) == 1 and isinstance(args[0], (list, tuple)) else args
    for i, net in enumerate(net_list):
      self.nets[f'{i}'] = net
    for name, net in kwargs.items():
      self.nets[name] = net

  def forward(self, *args, **kwargs):
    parts = [net(*args, **kwargs) for net in self.nets.values()]

    return torch.cat(parts dim=self.cat_dim) if dim=self.cat_dim is not None else parts

