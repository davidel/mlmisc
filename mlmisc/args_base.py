import collections

import py_misc_utils.assert_checks as tas
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

  for net in net_args:
    if isinstance(net, (list, tuple)):
      tas.check_eq(len(net), 2,
                   msg=f'In case of list/tuple argument, it must contain two elements')
      name, enet = net

      tas.check(name not in modules, msg=f'Module "{name}" already present')
      modules[name] = enet
    else:
      modules[f'{len(modules)}'] = net
  for name, net in kwargs.items():
    tas.check(name not in modules, msg=f'Module "{name}" already present')
    modules[name] = net

  return modules


class ArgsBase(nn.ModuleDict):

  def __init__(self, *args, **kwargs):
    super().__init__(expand_modules(args, kwargs))

