import itertools
import math

import numpy as np
import py_misc_utils.alog as alog
import py_misc_utils.assert_checks as tas
import py_misc_utils.utils as pyu
import torch
import torch.nn as nn

from . import args_sequential as aseq
from . import einops_layers as eil
from . import layer_utils as lu
from . import utils as ut


def calc_shape(n, c):
  s = int(math.sqrt(n / c))
  q = s * s
  k = math.ceil(n / q)

  return (k, s, s), q * k - n


def calc_best_shape(flat_size, base_c, min_dim):
  shape, pad = None, None
  for c in itertools.count(base_c):
    cshape, cpad = calc_shape(flat_size, c)
    if cshape[-1] < min_dim:
      break
    if shape is None or cpad < pad:
      shape, pad = cshape, cpad

  return shape, pad


def calc_conv_params(shape, out_features):
  stride, kernel_size, out_channels, error = None, None, None, None
  for cstride in itertools.count(1):
    ckernel_size = 2 * cstride + 1
    if ckernel_size >= shape[-1] // 2:
      break

    out_wnd_size = int((shape[-1] - ckernel_size) / cstride + 1)
    cout_channels = round(out_features / out_wnd_size**2)
    cerror = abs(out_features - cout_channels * out_wnd_size**2)
    if error is None or cerror < error:
      stride, kernel_size, out_channels = cstride, ckernel_size, cout_channels
      error = cerror

  alog.debug(f'Conv params: stride={stride} kernel_size={kernel_size} ' \
             f'out_wnd_size={out_wnd_size} out_channels={out_channels}')

  return kernel_size, stride, out_channels


def create_conv(in_channels, out_channels, kernel_size, stride,
                dropout, act):
  layers = [
    nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding='valid'),
    eil.Rearrange('b c h w -> b (c h w)'),
  ]
  if dropout is not None:
    layers.append(nn.Dropout(dropout))
  if act is not None:
    layers.append(lu.create(act))

  return aseq.ArgsSequential(layers)


class ConvLinear(nn.Module):

  def __init__(self, in_features, out_features,
               base_channels=None,
               min_dim_size=None,
               num_convs=None,
               dropout=None,
               act=None):
    base_channels = base_channels or 2
    min_dim_size = min_dim_size or 5
    num_convs = num_convs or 8

    shape, pad = calc_best_shape(in_features, base_channels, min_dim_size)

    alog.debug(f'Input reshape at {shape} with {pad} padding, for {in_features} -> {out_features} linear')

    tas.check_is_not_none(shape, msg=f'ConvLinear not supported for input size {in_features}')

    kernel_size, stride, out_channels = calc_conv_params(shape, out_features)

    convs = [create_conv(shape[0], out_channels, kernel_size, stride, dropout, act)
             for _ in range(num_convs)]

    super().__init__()
    self.shape, self.pad = shape, pad
    self.convs = nn.ModuleList(convs)

  def forward(self, x):
    lpad = self.pad // 2
    rpad = self.pad - lpad
    y = nn.functional.pad(x, (lpad, rpad))
    y = y.reshape(x.shape[0], *self.shape)
    y = ut.add([conv(y) for conv in self.convs])

    return y

