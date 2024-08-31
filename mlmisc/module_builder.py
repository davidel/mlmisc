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
    self.shape = ut.net_shape(net, self.shape)
    self.layers.append(net)
    self.config.append(NetConfig(input_fn=input_fn,
                                 output_fn=output_fn,
                                 net_args=net_args))

    return len(self.layers) - 1

  def linear(self, odim, input_fn=None, output_fn=None, **kwargs):
    return self.add(nn.Linear(self.shape[-1], odim, **kwargs),
                    input_fn=input_fn,
                    output_fn=output_fn)

  def conv2d(self, odim, input_fn=None, output_fn=None, **kwargs):
    return self.add(nn.Conv2d(self.shape[-3], odim, **kwargs),
                    input_fn=input_fn,
                    output_fn=output_fn)

  def deconv2d(self, odim, input_fn=None, output_fn=None, **kwargs):
    return self.add(nn.ConvTranspose2d(self.shape[-3], odim, **kwargs),
                    input_fn=input_fn,
                    output_fn=output_fn)

  def batchnorm2d(self, input_fn=None, output_fn=None, **kwargs):
    return self.add(nn.BatchNorm2d(self.shape[-3], **kwargs),
                    input_fn=input_fn,
                    output_fn=output_fn)

  def layernorm(self, input_fn=None, output_fn=None, **kwargs):
    return self.add(nn.LayerNorm(self.shape[-1], **kwargs),
                    input_fn=input_fn,
                    output_fn=output_fn)

  def forward(self, x, **kwargs):
    y, results = x, []
    for i, (net, cfg) in enumerate(zip(self.layers, self.config)):
      net_kwargs = dict()
      if cfg.input_fn is None:
        xx = (y,)
      else:
        xx = cfg.input_fn(y, results)
        if isinstance(xx, dict):
          net_kwargs.update(xx['kwargs'])
          xx = xx['args']

        xx = pyu.as_sequence(xx)

      for k in cfg.net_args:
        net_kwargs[k] = kwargs.get(k)

      res = net(*xx, **net_kwargs)

      results.append(res)
      y = res if cfg.output_fn is None else cfg.output_fn(res)

    return y

