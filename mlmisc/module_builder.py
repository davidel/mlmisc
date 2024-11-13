import collections

import py_misc_utils.utils as pyu
import torch
import torch.nn as nn

from . import nets_dict as netd
from . import utils as ut


NetConfig = collections.namedtuple(
  'NetConfig',
  'input_fn, output_fn, net_args'
)

_ADD_FIELDS = NetConfig._fields + ('in_shapes',)

class ModuleBuilder(nn.Module):

  def __init__(self, shape):
    super().__init__()
    self.shape = tuple(shape)
    self.layers = netd.NetsDict()
    self.config = []
    self.aux_modules = netd.NetsDict()

  def _pop_add_args(self, kwargs):
    args = pyu.pop_kwargs(kwargs, _ADD_FIELDS)

    return {name: value for name, value in zip(_ADD_FIELDS, args)}

  def __len__(self):
    return len(self.layers)

  def last_id(self):
    return len(self.layers) - 1

  def add(self, net,
          input_fn=None,
          output_fn=None,
          net_args=None,
          in_shapes=None):
    # The shape contains no batch dimension!
    if in_shapes is None:
      self.shape = ut.net_shape(net, self.shape)
    else:
      self.shape = ut.net_shape(net, *in_shapes)
    self.layers.add_net(net)
    self.config.append(NetConfig(input_fn=input_fn,
                                 output_fn=output_fn,
                                 net_args=net_args))
    # If the input/output functions are modules, store them here so that their
    # parameters can then be saved/loaded from the normal PyTorch state-dict machinery.
    if isinstance(input_fn, nn.Module):
      self.aux_modules.add_net(input_fn)
    if isinstance(output_fn, nn.Module):
      self.aux_modules.add_net(output_fn)

    return len(self.layers) - 1

  def linear(self, nout, **kwargs):
    aargs = self._pop_add_args(kwargs)
    return self.add(nn.Linear(self.shape[-1], nout, **kwargs), **aargs)

  def conv1d(self, nout, kernel_size, **kwargs):
    aargs = self._pop_add_args(kwargs)
    return self.add(nn.Conv1d(self.shape[-2], nout, kernel_size, **kwargs), **aargs)

  def deconv1d(self, nout, kernel_size, **kwargs):
    aargs = self._pop_add_args(kwargs)
    return self.add(nn.ConvTranspose1d(self.shape[-2], nout, kernel_size, **kwargs), **aargs)

  def conv2d(self, nout, kernel_size, **kwargs):
    aargs = self._pop_add_args(kwargs)
    return self.add(nn.Conv2d(self.shape[-3], nout, kernel_size, **kwargs), **aargs)

  def deconv2d(self, nout, kernel_size, **kwargs):
    aargs = self._pop_add_args(kwargs)
    return self.add(nn.ConvTranspose2d(self.shape[-3], nout, kernel_size, **kwargs), **aargs)

  def conv3d(self, nout, kernel_size, **kwargs):
    aargs = self._pop_add_args(kwargs)
    return self.add(nn.Conv3d(self.shape[-4], nout, kernel_size, **kwargs), **aargs)

  def deconv3d(self, nout, kernel_size, **kwargs):
    aargs = self._pop_add_args(kwargs)
    return self.add(nn.ConvTranspose3d(self.shape[-4], nout, kernel_size, **kwargs), **aargs)

  def batchnorm2d(self, **kwargs):
    aargs = self._pop_add_args(kwargs)
    return self.add(nn.BatchNorm2d(self.shape[-3], **kwargs), **aargs)

  def batchnorm1d(self, **kwargs):
    aargs = self._pop_add_args(kwargs)
    return self.add(nn.BatchNorm1d(self.shape[-2], **kwargs), **aargs)

  def layernorm(self, ndims=1, **kwargs):
    aargs = self._pop_add_args(kwargs)
    return self.add(nn.LayerNorm(self.shape[-ndims:], **kwargs), **aargs)

  def forward(self, *args, **kwargs):
    y, results = args, []
    for net, cfg in zip(self.layers.values(), self.config):
      net_kwargs = dict()
      if cfg.input_fn is None:
        xx = y
      else:
        xx = cfg.input_fn(y, results)
        if isinstance(xx, dict):
          net_kwargs.update(xx.get('kwargs', dict()))
          xx = xx.get('args', xx)

      xx = pyu.as_sequence(xx)

      for k in cfg.net_args or ():
        nk, wk = k if isinstance(k, (list, tuple)) else (k, k)
        net_kwargs[nk] = kwargs.get(wk)

      res = net(*xx, **net_kwargs)

      results.append(res)
      y = res if cfg.output_fn is None else cfg.output_fn(res)

    return y


def inputfn(result_ids, back=2):
  iid = len(result_ids) - back
  rid = result_ids[iid] if iid >= 0 else None

  def input_fn(x, results):
    return (x + results[rid]) if rid is not None else x

  return input_fn


def inputsum(rid):

  def input_fn(x, results):
    return x + results[rid]

  return input_fn

