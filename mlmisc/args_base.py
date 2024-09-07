import collections

import torch
import torch.nn as nn


def expand_modules(args, kwargs):
  modules, net_args = collections.OrderedDict(), args
  if len(args) == 1:
    arg = args[0]
    if isinstance(arg, (dict, collections.OrderedDict, nn.ModuleDict)):
      for name, net in arg.items():
        modules[name] = net
      net_args = []
    elif isinstance(arg, (list, tuple)):
      net_args = arg

  for i, net in enumerate(net_args):
    modules[f'{i}'] = net
  for name, net in kwargs.items():
    modules[name] = net

  return modules


class ArgsBase(nn.ModuleDict):

  def __init__(self, *args, **kwargs):
    super().__init__(expand_modules(args, kwargs))

