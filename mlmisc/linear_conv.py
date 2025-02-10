import math
import types

import numpy as np
import py_misc_utils.alog as alog
import torch
import torch.nn as nn

from . import conv_utils as cvu
from . import layer_utils as lu
from . import nn_lambda as lm
from . import module_builder as mb


def _get_wnd_size(size, kernel_size, num_layers, stride=None, dilation=None):
  if stride is None:
    stride = [1] * num_layers
  if dilation is None:
    dilation = [1] * num_layers
  for i in range(num_layers):
    size = cvu.deconv_wndsize(size, kernel_size, stride[i], dilation=dilation[i])

  return size


def _compute_linconv_convs(in_features,
                           out_features,
                           kernel_size,
                           num_layers,
                           compression):
  in_shape = cvu.squarest_shape(in_features)
  shape = tuple(_get_wnd_size(s, kernel_size, num_layers) for s in in_shape)
  out_channels = math.ceil(out_features / np.prod(shape))
  mid_channels = int(math.sqrt(compression * (in_features * out_features) /
                               (num_layers * kernel_size**2)))

  alog.debug(f'In: {in_features}, Out: {out_features}, Ksize: {kernel_size}, ' \
             f'Layers: {num_layers}')
  alog.debug(f'InShape: {in_shape}, Shape: {shape}, OutChan: {out_channels}, ' \
             f'MidChan: {mid_channels}')

  return types.SimpleNamespace(out_channels=out_channels,
                               mid_channels=mid_channels,
                               out_features=out_channels * np.prod(shape),
                               shape=shape,
                               in_shape=in_shape)


def _input_reshape(x, shape=None):
  new_shape = tuple(x.shape[: -1]) + (1,) + tuple(shape)

  return torch.reshape(x, new_shape)


def _select_flatten(x, out_features=None):
  y = torch.reshape(x, tuple(x.shape[: -3]) + (-1,))

  return y[..., : out_features]


def _build_linconv_net(in_features,
                       out_features,
                       kernel_size,
                       num_layers,
                       act,
                       compression,
                       reducer_kernel_size=3):
  lxres = _compute_linconv_convs(in_features,
                                 out_features,
                                 kernel_size,
                                 num_layers,
                                 compression)

  net = mb.ModuleBuilder((in_features,))
  net.add(lm.Lambda(_input_reshape, shape=lxres.in_shape, _info='Input Reshape'))
  for i in range(num_layers):
    net.deconv2d(lxres.mid_channels, kernel_size)
    net.batchnorm2d()
    net.add(lu.create(act))

  net.conv2d(lxres.out_channels, reducer_kernel_size, padding='same')

  net.add(lm.Lambda(_select_flatten,
                    out_features=out_features,
                    _info='Flatten Select'))

  return net


class LinearConv(nn.Module):

  def __init__(self, in_features, out_features, kernel_size, num_layers,
               act='relu',
               compression=0.05):
    super().__init__()
    self.net = _build_linconv_net(in_features,
                                  out_features,
                                  kernel_size,
                                  num_layers,
                                  act,
                                  compression)

  def forward(self, x):
    if x.ndim <= 2:
      return self.net(x)
    else:
      ishape = x.shape[: -1]

      y = torch.reshape(x, (-1, x.shape[-1]))
      y = self.net(y)

      return torch.reshape(y, ishape + y.shape[-1:])

