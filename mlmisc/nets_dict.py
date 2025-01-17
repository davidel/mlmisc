import collections

import py_misc_utils.assert_checks as tas
import py_misc_utils.core_utils as pycu
import py_misc_utils.utils as pyu
import torch
import torch.nn as nn


class NetsDict(nn.ModuleDict):

  def __init__(self, *args, **kwargs):
    super().__init__()
    self._name_gen = pycu.RevGen()
    self._expand_modules(args, kwargs)

  def _net_name(self, net):
    cls = pyu.cname(net)
    while True:
      name = self._name_gen.newname(cls)
      if name not in self:
        return name

  def _expand_modules(self, args, kwargs):
    net_args = args
    if len(args) == 1:
      arg = args[0]
      if pycu.isdict(arg) or isinstance(arg, nn.ModuleDict):
        for name, net in arg.items():
          self.add_net(net, name=name)
        net_args = []
      elif isinstance(arg, (list, tuple, nn.ModuleList)):
        net_args = arg

    for net in net_args:
      if isinstance(net, (list, tuple)):
        tas.check_eq(len(net), 2,
                     msg=f'In case of list/tuple argument, it must contain two elements')
        name, net = net

        self.add_net(net, name=name)
      else:
        self.add_net(net)
    for name, net in kwargs.items():
      self.add_net(net, name=name)

  def add_net(self, net, name=None, replace=False):
    if name is not None:
      tas.check(replace or name not in self,
                msg=f'Module "{name}" ({pyu.cname(net)}) already present')
      self[name] = net
    else:
      name = self._net_name(net)
      self[name] = net

