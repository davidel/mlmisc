import math

import py_misc_utils.alog as alog
import py_misc_utils.assert_checks as tas
import torch
import torch.nn as nn

from . import args_sequential as aseq
from . import conv_utils as cvu
from . import core_utils as cu
from . import einops_layers as eil
from . import layer_utils as lu
from . import types as typ


def calc_shape(n, c):
  s = math.ceil(math.sqrt(n / c))

  return typ.Shape2d(c, s, s), c * s**2 - n


def calc_best_shape(flat_size, in_channels, min_dim):
  shape, pad = None, None
  for c in in_channels:
    cshape, cpad = calc_shape(flat_size, c)
    if cshape.w < min_dim and shape is not None:
      break
    if shape is None or cpad < pad:
      shape, pad = cshape, cpad

  return shape, pad


def create_conv(shape, out_channels, kernel_size, stride, out_features, force):
  layers = [
    nn.Conv2d(shape.c, out_channels, kernel_size=kernel_size, stride=stride, padding='valid'),
    eil.Rearrange('b c h w -> b (c h w)'),
  ]
  flat_size = out_channels * cvu.conv_wndsize(shape.w, kernel_size, stride)**2
  if force:
    if flat_size != out_features:
      layers.append(nn.Linear(flat_size, out_features, bias=False))
      flat_size = out_features

  return aseq.ArgsSequential(layers), flat_size


class ConvLinear(nn.Module):

  def __init__(self, in_features, out_features,
               in_channels=tuple(range(1, 4)),
               min_dim_size=6,
               bias=True,
               force=False):
    shape, pad = calc_best_shape(in_features, in_channels, min_dim_size)
    conv_params = cvu.conv_flat_reduce(shape, out_features, force=force)
    tas.check_is_not_none(conv_params,
                          msg=f'ConvLinear not supported for {in_features} -> {out_features}')

    conv, flat_size = create_conv(
      shape,
      conv_params.channels,
      conv_params.kernel_size,
      conv_params.stride,
      out_features,
      force)

    alog.debug(f'Input reshape at {shape} with {pad} padding, for {in_features} -> ' \
               f'{out_features} ({flat_size})')

    super().__init__()
    self.shape, self.pad = shape, pad
    self.conv = conv
    self.bias = nn.Parameter(torch.zeros(flat_size)) if bias else None

  def forward(self, x):
    if self.pad:
      lpad = self.pad // 2
      rpad = self.pad - lpad
      y = nn.functional.pad(x, (lpad, rpad))
    else:
      y = x

    y = y.reshape(y.shape[0], *self.shape)
    y = self.conv(y)
    if self.bias is not None:
      y = y + self.bias

    return y

  def extra_repr(self):
    return cu.extra_repr(shape=self.shape, pad=self.pad)

