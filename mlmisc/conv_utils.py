import collections
import itertools
import math
import random

import py_misc_utils.alog as alog
import py_misc_utils.assert_checks as tas
import py_misc_utils.core_utils as pycu
import py_misc_utils.num_utils as pynu
import py_misc_utils.rnd_utils as pyr
import py_misc_utils.utils as pyu
import numpy as np
import torch
import torch.nn as nn

from . import layer_utils as lu
from . import module_builder as mb
from . import types as typ
from . import utils as ut


ConvSpec = collections.namedtuple(
  'ConvSpec',
  'features, kernel_size, stride, padding, maxpool, avgpool, norm, act',
  defaults=(1, 'valid', None, None, True, 'relu'),
)


def apply_conv_spec(net, cs):
  if cs.norm:
    net.batchnorm2d()
  net.conv2d(cs.features, cs.kernel_size,
             stride=cs.stride,
             padding=cs.padding)
  if cs.act is not None:
    net.add(lu.create(cs.act))
  if cs.maxpool is not None:
    net.add(nn.MaxPool2d(cs.maxpool))
  if cs.avgpool is not None:
    net.add(nn.AvgPool2d(cs.avgpool))


def build_conv_stack(convs, net=None, shape=None):
  net = net or mb.ModuleBuilder(shape)
  for cs in convs:
    apply_conv_spec(net, cs)

  alog.debug(f'ConvStack exit shape: {net.shape}')

  return net


CONVS_KEY = 'convs'

def load_conv_specs(path):
  cfg = pyu.load_config(path)
  alog.debug(f'Conv Specs Config:\n{pyu.config_to_string(cfg)}')

  conv_specs = []
  for cgroup in cfg[CONVS_KEY]:
    convs = tuple(ConvSpec(**cspec) for cspec in cgroup)
    conv_specs.append(convs)

  return conv_specs


def save_conv_specs(conv_specs, path):
  specs = []
  for convs in conv_specs:
    specs.append([cs._asdict() for cs in convs])

  cfg = {CONVS_KEY: specs}
  pyu.write_config(cfg, path)


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
                        round_features=16,
                        fsigma=0.2,
                        act=None,
                        tail=None,
                        **kwargs):
  net = net or mb.ModuleBuilder(shape)

  kernel_values, kernel_weights = _load_params(
    kwargs,
    'kernel',
    (2, 3, 4, 5, 6, 7, 8, 9),
    (1, 7, 1, 5, 1, 3, 1, 2)
  )
  stride_values, stride_weights = _load_params(
    kwargs,
    'stride',
    (1, 2, 3, 4),
    (7, 5, 1, 1)
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
    cshape = typ.Shape2d(*net.shape)

    min_size = min(cshape.h, cshape.w)
    if min_size < 3:
      ksize, stride, features = (cshape.h, cshape.w), 1, max_output
      padding = 'valid'
      maxpool, avgpool = None, None
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

      maxpool, avgpool = None, None
      if stride == 1:
        pool = pool_type_values[pyr.choices(pool_type_weights, 1)[0]]
        if pool is not None:
          pool_size = min(pool_values[pyr.choices(pool_weights, 1)[0]],
                          (min_size - 2 * ksize) // stride)
          if pool == 'max':
            maxpool = pool_size if pool_size > 1 else maxpool
          elif pool == 'avg':
            avgpool = pool_size if pool_size > 1 else avgpool

      norm = norm_values[pyr.choices(norm_weights, 1)[0]]

      features = int(cshape.c * max(1.0, random.normalvariate(mu=float(stride),
                                                              sigma=fsigma)))
      features = pynu.round_up(features, round_features)

    opt_args = pycu.denone(maxpool=maxpool, avgpool=avgpool, act=act)

    convs.append(ConvSpec(features=features,
                          kernel_size=ksize,
                          stride=stride,
                          padding=padding,
                          norm=norm,
                          **opt_args))

    alog.debug(f'Layer: {convs[-1]}')

    apply_conv_spec(net, convs[-1])

  if tail is not None:
    net = tail(net)

  return net, tuple(convs)


CONVSPEC_ARGMAP = {
  'f': 'features',
  'k': 'kernel_size',
  's': 'stride',
  'p': 'padding',
  'x': 'maxpool',
  'v': 'avgpool',
  'n': 'norm',
  'a': 'act',
}

def convs_from_string(config, defaults=None):
  convs = []
  for conv in pyu.resplit(config, ':'):
    conv_args = defaults.copy() if defaults else dict()
    conv_args.update({CONVSPEC_ARGMAP.get(k, k): v
                      for k, v in pyu.parse_dict(conv).items()})
    convs.append(ConvSpec(**conv_args))

  return convs


def conv_wndsize(size, kernel_size, stride):
  return int((size - kernel_size) / stride + 1)


ReduceConvParams = collections.namedtuple(
  'ReduceConvParams',
  'error, stride, kernel_size, channels, wndsize')

def conv_flat_reduce(shape, out_features, force=False):
  shape = typ.Shape2d(*shape)
  min_size = min(shape.h, shape.w)
  params = []
  for stride in itertools.count(1):
    param_count = len(params)
    for channels in itertools.count(1):
      kernel_size = int(math.sqrt(out_features / channels))
      kernel_size = min(min_size, kernel_size)
      if kernel_size < 2 * stride + 1:
        break

      wndsize_h = conv_wndsize(shape.h, kernel_size, stride)
      wndsize_w = conv_wndsize(shape.w, kernel_size, stride)
      wndsize = (wndsize_h, wndsize_w)
      error = channels * wndsize_h * wndsize_w - out_features
      params.append(ReduceConvParams(error, stride, kernel_size, channels, wndsize))

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

