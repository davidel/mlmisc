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

