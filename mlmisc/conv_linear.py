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
  s = math.ceil(math.sqrt(n / c))

  return (c, s, s), c * s**2 - n


def calc_best_shape(flat_size, max_channels, min_dim):
  shape, pad = None, None
  for c in range(1, max_channels + 1):
    cshape, cpad = calc_shape(flat_size, c)
    if cshape[-1] < min_dim and shape is not None:
      break
    if shape is None or cpad < pad:
      shape, pad = cshape, cpad

  return shape, pad


def conv_wndsize(size, kernel_size, stride):
  return int((size - kernel_size) / stride + 1)


def calc_conv_params(shape, out_features, force):
  params = []
  for stride in itertools.count(1):
    kernel_size = 2 * stride + 1
    if kernel_size > shape[-1] // 2:
      break

    wndsize = conv_wndsize(shape[-1], kernel_size, stride)
    channels = max(1, round(out_features / wndsize**2))
    error = channels * wndsize**2 - out_features

    params.append((error, stride, kernel_size, channels, wndsize))

  if params:
    if force:
      # When "forcing" we are going to add a marshaling linear layer, so it is better
      # to end up with an higher flattened size, and turn it down, instead of the
      # contrary. Hence we look for "error" >= 0, if any. otherwise we pick the less
      # negative error.
      params = sorted(params, key=lambda x: x[0])
      best_param = params[-1]
      for p in params:
        if p[0] >= 0:
          best_param = p
          break

      stride, kernel_size, channels, wndsize = best_param[1:]
    else:
      params = sorted(params, key=lambda x: abs(x[0]))
      stride, kernel_size, channels, wndsize = params[0][1:]

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
    flat_size = out_channels * conv_wndsize(shape[-1], kernel_size, stride)**2
    if flat_size != out_features:
      layers.append(nn.Linear(flat_size, out_features, bias=False))

  if dropout is not None:
    layers.append(nn.Dropout(dropout))
  if act is not None:
    layers.append(lu.create(act))

  return aseq.ArgsSequential(layers)


class ConvLinear(nn.Module):

  def __init__(self, in_features, out_features,
               max_channels=None,
               min_dim_size=None,
               dropout=None,
               act=None,
               force=None):
    max_channels = max_channels or 3
    min_dim_size = min_dim_size or 6
    force = False if force is None else True

    shape, pad = calc_best_shape(in_features, max_channels, min_dim_size)

    alog.debug(f'Input reshape at {shape} with {pad} padding, for {in_features} -> {out_features}')

    conv_params = calc_conv_params(shape, out_features, force)

    tas.check_is_not_none(conv_params,
                          msg=f'ConvLinear not supported for {in_features} -> {out_features}')
    kernel_size, stride, out_channels = conv_params

    super().__init__()
    self.shape, self.pad = shape, pad
    self.conv = create_conv(shape, out_channels, kernel_size, stride, out_features,
                            force, dropout, act)

  def forward(self, x):
    lpad = self.pad // 2
    rpad = self.pad - lpad
    y = nn.functional.pad(x, (lpad, rpad))
    y = y.reshape(y.shape[0], *self.shape)
    y = self.conv(y)

    return y

  def extra_repr(self):
    return pyu.stri(dict(shape=self.shape, pad=self.pad))

