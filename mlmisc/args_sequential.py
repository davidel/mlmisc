import py_misc_utils.inspect_utils as pyiu
import py_misc_utils.utils as pyu
import torch
import torch.nn as nn

from . import nets_dict as netd


# Example usage of Wrap() and Mix() with ArgsSequential():
#
#   stack = ArgsSequential(
#     nn.Linear(num_states, hid_size),
#     l1 := Wrap(nn.LayerNorm(hid_size)),
#     lu.create(act),
#     nn.Linear(hid_size, hid_size),
#     l2 := Wrap(nn.LayerNorm(hid_size)),
#     lu.create(act),
#     nn.Linear(hid_size, hid_size),
#     nn.LayerNorm(hid_size),
#     Mix(lambda *x: sum(x), l1, l2),
#     ...
#   )
#
class Wrap(nn.Module):

  def __init__(self, net):
    super().__init__()
    self.net = net
    self.res = None

  def forward(self, *args, **kwargs):
    self.res = self.net(*args, **kwargs)

    return self.res


class Mix(nn.Module):

  def __init__(self, func, *nets):
    super().__init__()
    self.func = func
    self.nets = nets
    self.res = None

  def forward(self, *args):
    fargs = args + tuple(net.res for net in self.nets)

    self.res = self.func(*fargs)

    return self.res

  def extra_repr(self):
    reprs = ',\n'.join(repr(net) for net in self.nets)

    return reprs


class ArgsSequential(netd.NetsDict):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._net_args = dict()

  def _get_arg_names(self, name, net):
    args = self._net_args.get(name)
    if args is None:
      if hasattr(net, 'forward'):
        args = pyiu.get_arg_names(net.forward, positional=False)
      else:
        args = ()

      self._net_args[name] = args

    return args

  def forward(self, x, *args, **kwargs):
    y = x
    for name, net in self.items():
      net_args = self._get_arg_names(name, net)
      net_kwargs = pyu.mget(kwargs, *net_args, as_dict=True)

      y = net(y, *args, **net_kwargs)

    return y

