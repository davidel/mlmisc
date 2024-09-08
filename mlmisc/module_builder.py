import collections

import py_misc_utils.utils as pyu
import torch
import torch.nn as nn

from . import utils as ut


NetConfig = collections.namedtuple(
  'NetConfig',
  'input_fn, output_fn, net_args'
)


class ModuleBuilder(nn.Module):

  def __init__(self, shape):
    super().__init__()
    self.shape = tuple(shape)
    self.layers = nn.ModuleList()
    self.config = []

  def add(self, net, input_fn=None, output_fn=None, net_args=()):
    # The shape contains no batch dimension!
    self.shape = ut.net_shape(net, self.shape)
    self.layers.append(net)
    self.config.append(NetConfig(input_fn=input_fn,
                                 output_fn=output_fn,
                                 net_args=net_args))

    return len(self.layers) - 1

  def linear(self, nout, args_={}, **kwargs):
    return self.add(nn.Linear(self.shape[-1], nout, **kwargs), **args_)

  def conv2d(self, nout, args_={}, **kwargs):
    return self.add(nn.Conv2d(self.shape[-3], nout, **kwargs), **args_)

  def deconv2d(self, nout, args_={}, **kwargs):
    return self.add(nn.ConvTranspose2d(self.shape[-3], nout, **kwargs), **args_)

  def batchnorm2d(self, args_={}, **kwargs):
    return self.add(nn.BatchNorm2d(self.shape[-3], **kwargs), **args_)

  def batchnorm1d(self, args_={}, **kwargs):
    return self.add(nn.BatchNorm1d(self.shape[0], **kwargs), **args_)

  def layernorm(self, ndims, args_={}, **kwargs):
    return self.add(nn.LayerNorm(self.shape[-ndims: ], **kwargs), **args_)

  def forward(self, *args, **kwargs):
    y, results = args, []
    for net, cfg in zip(self.layers, self.config):
      net_kwargs = dict()
      if cfg.input_fn is None:
        xx = y
      else:
        xx = cfg.input_fn(y, results)
        if isinstance(xx, dict):
          net_kwargs.update(xx.get('kwargs', dict()))
          xx = xx['args']

      xx = pyu.as_sequence(xx)

      for k in cfg.net_args:
        net_kwargs[k] = kwargs.get(k)

      res = net(*xx, **net_kwargs)

      results.append(res)
      y = res if cfg.output_fn is None else cfg.output_fn(res)

    return y

