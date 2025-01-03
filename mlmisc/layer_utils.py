import py_misc_utils.alog as alog
import py_misc_utils.utils as pyu
import torch
import torch.nn as nn

from . import auto_module as am
from . import config as conf


_LAYERS = {
  'elu': nn.ELU,
  'gelu': nn.GELU,
  'hswish': nn.Hardswish,
  'id': nn.Identity,
  'leay_relu': nn.LeakyReLU,
  'mish': nn.Mish,
  'prelu': nn.PReLU,
  'relu': nn.ReLU,
  'silu': nn.SiLU,
  'selu': nn.SELU,
  'sigm': nn.Sigmoid,
  'tanh': nn.Tanh,
}

def create(lay):
  if isinstance(lay, str):
    cls = _LAYERS.get(lay)

    return cls() if cls is not None else conf.create_object('Layer', lay)
  elif isinstance(lay, nn.Module):
    if am.is_auto(lay):
      return am.new_as(lay)

    alog.xraise(ValueError, f'Module object not accept since cannot be replicated ' \
                f'(use auto_module.create() to create an instance): {lay}')
  elif callable(lay):
    return lay()
  else:
    alog.xraise(ValueError, f'Unknown layer specification: {lay}')


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
    fargs = list(args) + [net.res for net in self.nets]

    self.res = self.func(*fargs)

    return self.res

  def extra_repr(self):
    reprs = ',\n'.join(repr(net) for net in self.nets)

    return reprs

