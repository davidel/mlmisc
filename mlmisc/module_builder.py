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

  ADD_ARGS = ('input_fn', 'output_fn', 'net_args')

  def __init__(self, shape):
    super().__init__()
    self.shape = tuple(shape)
    self.layers = nn.ModuleList()
    self.config = []

  def _pop_add_args(self, kwargs):
    args = pyu.pop_kwargs(kwargs, self.ADD_ARGS)

    return {name: value for name, value in zip(self.ADD_ARGS, args)}

  def __len__(self):
    return len(self.layers)

  def last_id(self):
    return len(self.layers) - 1

  def add(self, net,
          input_fn=None,
          output_fn=None,
          net_args=None):
    # The shape contains no batch dimension!
    self.shape = ut.net_shape(net, self.shape)
    self.layers.append(net)
    self.config.append(NetConfig(input_fn=input_fn,
                                 output_fn=output_fn,
                                 net_args=net_args))

    return len(self.layers) - 1

  def linear(self, nout, **kwargs):
    aargs = self._pop_add_args(kwargs)
    return self.add(nn.Linear(self.shape[-1], nout, **kwargs), **aargs)

  def conv2d(self, nout, **kwargs):
    aargs = self._pop_add_args(kwargs)
    return self.add(nn.Conv2d(self.shape[-3], nout, **kwargs), **aargs)

  def deconv2d(self, nout, **kwargs):
    aargs = self._pop_add_args(kwargs)
    return self.add(nn.ConvTranspose2d(self.shape[-3], nout, **kwargs), **aargs)

  def conv3d(self, nout, **kwargs):
    aargs = self._pop_add_args(kwargs)
    return self.add(nn.Conv3d(self.shape[-4], nout, **kwargs), **aargs)

  def deconv3d(self, nout, **kwargs):
    aargs = self._pop_add_args(kwargs)
    return self.add(nn.ConvTranspose3d(self.shape[-4], nout, **kwargs), **aargs)

  def batchnorm2d(self, **kwargs):
    aargs = self._pop_add_args(kwargs)
    return self.add(nn.BatchNorm2d(self.shape[-3], **kwargs), **aargs)

  def batchnorm1d(self, **kwargs):
    aargs = self._pop_add_args(kwargs)
    return self.add(nn.BatchNorm1d(self.shape[0], **kwargs), **aargs)

  def layernorm(self, ndims, **kwargs):
    aargs = self._pop_add_args(kwargs)
    return self.add(nn.LayerNorm(self.shape[-ndims: ], **kwargs), **aargs)

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

      for k in cfg.net_args or ():
        net_kwargs[k] = kwargs.get(k)

      res = net(*xx, **net_kwargs)

      results.append(res)
      y = res if cfg.output_fn is None else cfg.output_fn(res)

    return y

