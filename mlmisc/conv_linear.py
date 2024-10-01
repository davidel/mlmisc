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


def conv_wndsize(size, kernel_size, stride):
  return int((size - kernel_size) / stride + 1)


def calc_conv_params(shape, out_features):
  stride, kernel_size, channels, wndsize, error = None, None, None, None, None
  for cstride in itertools.count(1):
    ckernel_size = 2 * cstride + 1
    if ckernel_size > shape[-1] // 2:
      break

    cwndsize = conv_wndsize(shape[-1], ckernel_size, cstride)
    cchannels = max(1, round(out_features / cwndsize**2))
    cerror = abs(out_features - cchannels * cwndsize**2)
    if error is None or cerror < error:
      stride, kernel_size, channels, wndsize = cstride, ckernel_size, cchannels, cwndsize
      error = cerror

  if error is not None:
    alog.debug(f'Conv params: stride={stride} kernel_size={kernel_size} ' \
               f'out_wnd_size={wndsize} out_channels={channels}')

    return kernel_size, stride, channels


def create_conv(shape, out_channels, kernel_size, stride, out_features,
                force, dropout, act):
  layers = [
    nn.Conv2d(shape[0], out_channels, kernel_size=kernel_size, stride=stride, padding='valid'),
    eil.Rearrange('b c h w -> b (c h w)'),
  ]
  if force:
    cwndsize = conv_wndsize(shape[-1], kernel_size, stride)
    flat_size = out_channels * wndsize**2
    if flat_size != out_features:
      layers.append(nn.Linear(flat_size, out_features, bias=False))

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
               act=None,
               force=None):
    base_channels = base_channels or 2
    min_dim_size = min_dim_size or 6
    num_convs = num_convs or 8
    force = False if force is None else True

    shape, pad = calc_best_shape(in_features, base_channels, min_dim_size)

    alog.debug(f'Input reshape at {shape} with {pad} padding, for {in_features} -> {out_features}')

    tas.check_is_not_none(shape, msg=f'ConvLinear not supported for {in_features} -> {out_features}')

    conv_params = calc_conv_params(shape, out_features)

    tas.check_is_not_none(conv_params,
                          msg=f'ConvLinear not supported for {in_features} -> {out_features}')
    kernel_size, stride, out_channels = conv_params

    convs = [create_conv(shape, out_channels, kernel_size, stride, out_features,
                         force, dropout, act)
             for _ in range(num_convs)]

    super().__init__()
    self.shape, self.pad = shape, pad
    self.convs = nn.ModuleList(convs)

  def forward(self, x):
    lpad = self.pad // 2
    rpad = self.pad - lpad
    y = nn.functional.pad(x, (lpad, rpad))
    y = y.reshape(y.shape[0], *self.shape)
    y = ut.add([conv(y) for conv in self.convs])

    return y

