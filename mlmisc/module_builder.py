import collections
import functools

import py_misc_utils.core_utils as pycu
import py_misc_utils.obj as obj
import py_misc_utils.utils as pyu
import torch
import torch.nn as nn

from . import core_utils as cu
from . import nets_dict as netd


class Result(obj.Obj):

  def reset(self):
    vars(self).clear()


class ResultsNamespace(list):

  def new(self):
    self.append(Result())

    return self[-1]

  def reset(self):
    for result in self:
      result.reset()


class Capture(nn.Module):

  def __init__(self, ns):
    super().__init__()
    self._ns = ns

  def forward(self, x):
    self._ns.y = x

    return x


NetConfig = collections.namedtuple('NetConfig', 'input_fn, net_args')

_ADD_FIELDS = NetConfig._fields + ('in_shapes',)

class ModuleBuilder(nn.Module):

  def __init__(self, shape):
    super().__init__()
    self.shape = tuple(shape)
    self.layers = netd.NetsDict()
    self._config = []
    self.resns = ResultsNamespace()

  def _pop_add_args(self, kwargs):
    args = pyu.pop_kwargs(kwargs, _ADD_FIELDS)

    return {name: value for name, value in zip(_ADD_FIELDS, args)}

  def __len__(self):
    return len(self.layers)

  def last_id(self):
    # IDs refers to the indices within the "results" stack (see forward() API below),
    # which has been populated with the input value as first entry.
    # So the effective index within "results" of the layer N-1 (indexed from 0) is N.
    return len(self.layers)

  def add(self, net,
          input_fn=None,
          net_args=None,
          in_shapes=None):
    # The shape contains no batch dimension!
    if in_shapes is None:
      self.shape = cu.net_shape(net, self.shape)
    else:
      self.shape = cu.net_shape(net, *in_shapes)
    self.layers.add_net(net)
    self._config.append(NetConfig(input_fn=input_fn, net_args=net_args))

    # See comment in the last_id() API above.
    return len(self.layers)

  def supported_kwargs(self):
    args = set()
    for cfg in self._config:
      for k in cfg.net_args or ():
        nk, wk = k if isinstance(k, (list, tuple)) else (k, k)
        args.add(wk)

    return tuple(sorted(args))

  def capture(self, ns):
    return self.add(Capture(ns))

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
    return self.add(nn.BatchNorm1d(self.shape[-1], **kwargs), **aargs)

  def layernorm(self, ndims=1, **kwargs):
    aargs = self._pop_add_args(kwargs)
    return self.add(nn.LayerNorm(self.shape[-ndims:], **kwargs), **aargs)

  def forward(self, *args, **kwargs):
    y = args[0] if len(args) == 1 else args
    results = [y]
    for net, cfg in zip(self.layers.values(), self._config):
      net_kwargs = dict()
      if cfg.input_fn is None:
        xx = y
      else:
        xx = cfg.input_fn(y, results)
        if pycu.isdict(xx):
          net_kwargs.update(xx.get('kwargs', dict()))
          xx = xx.get('args', xx)

      xx = pyu.as_sequence(xx)

      for k in cfg.net_args or ():
        nk, wk = k if isinstance(k, (list, tuple)) else (k, k)
        net_kwargs[nk] = kwargs.get(wk)

      y = net(*xx, **net_kwargs)

      results.append(y)

    self.resns.reset()

    return y


def _inputsum(rid, x, results):
  return x + results[rid]


def inputsum(rid):
  return functools.partial(_inputsum, rid)


def inputsum_back(result_ids, back=2):
  iid = len(result_ids) - back

  return functools.partial(_inputsum, result_ids[iid]) if iid >= 0 else None


def _inputtuple(rid, x, results):
  return x, results[rid]


def inputtuple(rid):
  return functools.partial(_inputtuple, rid)


def _select(rid, x, results):
  return results[rid]


def select(rid):
  return functools.partial(_select, rid)

