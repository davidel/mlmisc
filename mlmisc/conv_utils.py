import collections
import random

import py_misc_utils.alog as alog
import py_misc_utils.assert_checks as tas
import py_misc_utils.rnd_utils as pyr
import py_misc_utils.utils as pyu
import numpy as np
import torch
import torch.nn as nn

from . import layer_utils as lu
from . import module_builder as mb
from . import utils as ut


ConvSpec = collections.namedtuple(
  'ConvSpec',
  'features, kernel_size, stride, padding, maxpool, avgpool, norm, act',
  defaults=(1, 'same', None, None, True, 'relu'),
)


def apply_conv_spec(net, cs, act=None):
  if cs.norm:
    net.batchnorm2d()
  net.conv2d(cs.features,
             kernel_size=cs.kernel_size,
             stride=cs.stride,
             padding=cs.padding)
  cact = act or cs.act
  if cact is not None:
    net.add(lu.create(cact))
  if cs.maxpool is not None:
    net.add(nn.MaxPool2d(cs.maxpool))
  if cs.avgpool is not None:
    net.add(nn.AvgPool2d(cs.avgpool))


def build_conv_stack(convs, net=None, shape=None, act=None):
  net = net or mb.ModuleBuilder(shape)
  for cs in convs:
    apply_conv_spec(net, cs, act=act)

  return net


def _load_params(kwargs, name, default_values, default_weights):
  pvalues = kwargs.get(f'{name}_values')
  if pvalues is not None and isinstance(pvalues, str):
    pvalues = tuple(pyu.infer_value(v) for v in pyu.comma_split(pvalues))
  if pvalues is None:
    pvalues = default_values

  pweights = kwargs.get(f'{name}_weights')
  if pweights is not None and isinstance(pweights, str):
    pweights = tuple(pyu.infer_value(v) for v in pyu.comma_split(pweights))
  if pweights is None:
    pweights = default_weights

  tas.check_eq(len(pvalues), len(pweights),
               msg=f'Values and weights lengths do not match')

  return pvalues, pweights


def create_random_stack(max_output,
                        net=None,
                        shape=None,
                        round_features=None,
                        act=None,
                        tail=None,
                        **kwargs):
  round_features = round_features or 16
  net = net or mb.ModuleBuilder(shape)

  kernel_values, kernel_weights = _load_params(
    kwargs,
    'kernel',
    (2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 1, 5, 1, 4, 1, 2)
  )
  stride_values, stride_weights = _load_params(
    kwargs,
    'stride',
    (1, 2, 3, 4),
    (6, 4, 1, 1)
  )
  padding_values, padding_weights = _load_params(
    kwargs,
    'padding',
    ('valid', 'same'),
    (3, 1)
  )
  pool_type_values, pool_type_weights = _load_params(
    kwargs,
    'pool_type',
    (None, 'avg', 'max'),
    (2, 1, 2)
  )
  pool_values, pool_weights = _load_params(
    kwargs,
    'pool',
    (2, 3, 4),
    (10, 2, 1)
  )
  norm_values, norm_weights = _load_params(
    kwargs,
    'norm',
    (True, False),
    (5, 1)
  )

  convs = []
  while np.prod(net.shape) > max_output:
    in_features, h, w = net.shape

    min_size = min(h, w)
    if min_size < 3:
      ksize, stride, features = (h, w), 1, max_output
      padding = 'valid'
      maxpool, avgpool = 0, 0
      norm = norm_values[pyr.choices(norm_weights, 1)[0]]
    else:
      ksize = kernel_values[pyr.choices(kernel_weights, 1)[0]]
      ksize = min(ksize, min_size)

      max_stride = max(1, ksize - 1)
      if max_stride > 1:
        while True:
          stride = stride_values[pyr.choices(stride_weights, 1)[0]]
          if stride <= max_stride:
            break
      else:
        stride = max_stride

      if stride != 1 or ksize % 2 == 0:
        padding = 'valid'
      else:
        padding = padding_values[pyr.choices(padding_weights, 1)[0]]

      maxpool, avgpool = 0, 0
      if stride == 1:
        pool = pool_type_values[pyr.choices(pool_type_weights, 1)[0]]
        if pool is not None:
          pool_size = min(pool_values[pyr.choices(pool_weights, 1)[0]],
                          (min_size - 2 * ksize) // stride)
          if pool == 'max':
            maxpool = pool_size
          elif pool == 'avg':
            avgpool = pool_size

      norm = norm_values[pyr.choices(norm_weights, 1)[0]]

      shrinkage = max(stride, maxpool, avgpool)
      features = int(in_features * max(1.0, random.normalvariate(mu=float(shrinkage), sigma=0.5)))

      features = pyu.round_up(features, round_features)

    convs.append(ConvSpec(features=features,
                          kernel_size=ksize,
                          stride=stride,
                          padding=padding,
                          maxpool=maxpool if maxpool > 1 else None,
                          avgpool=avgpool if avgpool > 1 else None,
                          norm=norm,
                          act=act))

    alog.debug(f'Layer: {convs[-1]}')

    apply_conv_spec(net, convs[-1])

  if tail is not None:
    tail(net)

  return net, tuple(convs)

