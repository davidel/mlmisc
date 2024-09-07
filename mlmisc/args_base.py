import collections

import torch
import torch.nn as nn


def expand_modules(args, kwargs):
  modules = collections.OrderedDict()
  net_args = args[0] if len(args) == 1 and isinstance(args[0], (list, tuple)) else args
  for i, net in enumerate(net_args):
    modules[f'{i}'] = net
  for name, net in kwargs.items():
    modules[name] = net

  return modules


class ArgsBase(nn.ModuleDict):

  def __init__(self, *args, **kwargs):
    super().__init__(expand_modules(args, kwargs))

