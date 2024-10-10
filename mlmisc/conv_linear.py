import collections
import itertools
import math

import numpy as np
import py_misc_utils.alog as alog
import py_misc_utils.assert_checks as tas
import py_misc_utils.utils as pyu
import torch
import torch.nn as nn

from . import args_sequential as aseq
from . import conv_utils as cu
from . import einops_layers as eil
from . import layer_utils as lu
from . import types as typ
from . import utils as ut


OutputConvParams = collections.namedtuple(
  'OutputConvParams',
  'error, stride, kernel_size, channels, wndsize')


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


def calc_conv_params(shape, out_features, max_params, force):
  params = []
  for stride in itertools.count(1):
    param_count = len(params)
    for channels in itertools.count(1):
      kernel_size = int(math.sqrt(max_params / (channels * shape.c)))
      kernel_size = min(shape.w, kernel_size)
      if kernel_size < 2 * stride + 1:
        break

      wndsize = cu.conv_wndsize(shape.w, kernel_size, stride)
      error = channels * wndsize**2 - out_features
      params.append(OutputConvParams(error, stride, kernel_size, channels, wndsize))

      alog.debug(f'Conv Params: error={error} stride={stride} kernel_size={kernel_size} ' \
                 f'out_wnd_size={wndsize} out_channels={channels}')

    if param_count == len(params):
      break

  if params:
    if force:
      # When "forcing" we are going to add a marshaling linear layer, so it is better
      # to end up with an higher flattened size, and turn it down, instead of the
      # contrary. Hence we look for "error" >= 0, if any. otherwise we pick the less
      # negative error.
      params = sorted(params, key=lambda x: x.error)
      best = params[-1]
      for p in params:
        if p.error >= 0:
          best = p
          break
    else:
      params = sorted(params, key=lambda x: abs(x.error))
      best = params[0]

    alog.debug(f'Selected: stride={best.stride} kernel_size={best.kernel_size} ' \
               f'out_wnd_size={best.wndsize} out_channels={best.channels}')

    return best


def create_conv(shape, out_channels, kernel_size, stride, out_features,
                force, dropout, act):
  layers = [
    nn.Conv2d(shape.c, out_channels, kernel_size=kernel_size, stride=stride, padding='valid'),
    eil.Rearrange('b c h w -> b (c h w)'),
  ]
  flat_size = out_channels * cu.conv_wndsize(shape.w, kernel_size, stride)**2
  if force:
    if flat_size != out_features:
      layers.append(nn.Linear(flat_size, out_features, bias=False))
      flat_size = out_features

  if dropout is not None:
    layers.append(nn.Dropout(dropout))
  if act is not None:
    layers.append(lu.create(act))

  return aseq.ArgsSequential(layers), flat_size


class ConvLinear(nn.Module):

  def __init__(self, in_features, out_features,
               params_reduction=None,
               in_channels=None,
               min_dim_size=None,
               dropout=None,
               bias=None,
               act=None,
               force=None):
    params_reduction = pyu.value_or(params_reduction, 0.2)
    in_channels = pyu.value_or(in_channels, range(1, 4))
    min_dim_size = pyu.value_or(min_dim_size, 6)
    bias = pyu.value_or(bias, True)
    force = pyu.value_or(force, False)

    shape, pad = calc_best_shape(in_features, in_channels, min_dim_size)
    max_params = round(in_features * out_features * params_reduction)
    conv_params = calc_conv_params(shape, out_features, max_params, force)
    tas.check_is_not_none(conv_params,
                          msg=f'ConvLinear not supported for {in_features} -> {out_features}')

    conv, flat_size = create_conv(
      shape,
      conv_params.channels,
      conv_params.kernel_size,
      conv_params.stride,
      out_features,
      force,
      dropout,
      act)

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
      y += self.bias

    return y

  def extra_repr(self):
    return pyu.stri(dict(shape=self.shape, pad=self.pad))

