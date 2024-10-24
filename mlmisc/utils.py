import functools
import importlib
import math
import operator
import os
import random
import re
import subprocess
import sys

import numpy as np
import py_misc_utils.alog as alog
import py_misc_utils.file_overwrite as pyfow
import py_misc_utils.module_utils as pymu
import py_misc_utils.utils as pyu
import torch
import torch.nn as nn
import torch.utils.tensorboard

from . import auto_module as am
from . import load_state_dict as lsd


class Training:

  def __init__(self, net, training):
    self.net = net
    self.training = training

  def __enter__(self):
    self.prev_training = self.net.training
    self.net.train(self.training)
    return self

  def __exit__(self, *exc):
    self.net.train(self.prev_training)
    return False


def get_device(kind=None):
  if kind is None:
    kind = 'cuda' if torch.cuda.is_available() else 'cpu'

  return torch.device(kind)


def randseed(seed):
  rseed = pyu.randseed(seed)
  torch.manual_seed(rseed)
  torch.cuda.manual_seed_all(rseed)

  return rseed


def item(v):
  fn = getattr(v, 'item', None)

  return fn() if fn is not None else v


def torch_dtype(dtype):
  return getattr(torch, dtype) if isinstance(dtype, str) else dtype


def is_integer(t):
  if isinstance(t, int):
    return True
  if isinstance(t, torch.Tensor):
    return not (t.is_floating_point() or t.is_complex())
  if isinstance(t, np.ndarray):
    return np.issubdtype(t.dtype, np.integer)

  return False


def torch_load(path, **kwargs):
  return torch.load(path, weights_only=False, **kwargs)


def torch_load_to(dest, path, **kwargs):
  alog.debug(f'Loading tensor data from {path} to tensor/parameter with ' \
             f'shape {tuple(dest.shape)} ...')
  t = torch.load(path, weights_only=True, **kwargs)
  dest.data.copy_(getattr(t, 'data', t))

  return dest


def net_shape(net, shape, device=None, dtype=None):
  with torch.no_grad(), Training(net, False):
    # Add and remove the artificial batch dimension.
    x = torch.randn((1,) + tuple(shape), dtype=dtype, device=device)
    y = net(x)

    return tuple(y.shape[1:])


def model_save(model, path):
  alog.debug(f'Saving model to {path} ...')
  with pyfow.FileOverwrite(path, mode='wb') as ptfd:
    torch.save(model.state_dict(), ptfd)
  alog.debug(f'Model saved to {path}')


def model_load(path, model=None, device=None, strict=None):
  map_location = torch.device('cpu') if device is not None else None
  if model is None:
    alog.debug(f'Loading model state from {path}')
    model = am.load(path, map_location=map_location, strict=strict)
  elif os.path.exists(path):
    alog.debug(f'Loading model state from {path}')
    lsd.load_state_dict(model,
                        torch.load(path, map_location=map_location, weights_only=True),
                        strict=strict)

  return model.to(device) if device is not None else model


def checkpoint_model(model, path, gs_path=None):
  model_save(model, path)

  if gs_path is not None:
    alog.debug(f'Copying model state to {gs_path}')
    subprocess.check_call(('gsutil',
                           '-m',
                           '-q',
                           'cp',
                           path,
                           gs_path))


def save_data(path, **kwargs):
  data = dict()
  for name, ndata in kwargs.items():
    sdfn = getattr(ndata, 'state_dict', None)
    if sdfn is not None and callable(sdfn):
      data[name] = sdfn()
    else:
      data[name] = ndata

  alog.debug(f'Saving data to {path} ...')
  with pyfow.FileOverwrite(path, mode='wb') as ptfd:
    torch.save(data, ptfd)
  alog.debug(f'Data saved to {path}')


def load_data(path, map_location=None, strict=None, **kwargs):
  alog.debug(f'Loading data from {path}')
  td = torch_load(path, map_location=map_location or torch.device('cpu'))

  data = dict()
  for name, ndata in td.items():
    sdobj = kwargs.get(name)
    if sdobj is not None:
      lsd.load_state_dict(sdobj, ndata, strict=strict)
      data[name] = sdobj
    elif isinstance(ndata, dict) and am.is_auto_state(ndata):
      data[name] = am.load(ndata, strict=strict)
    else:
      data[name] = ndata

  return data


def checkpoint_data(path, gs_path=None, **kwargs):
  save_data(path, **kwargs)

  if gs_path is not None:
    alog.debug(f'Copying data to {gs_path}')
    subprocess.check_call(('gsutil',
                           '-m',
                           '-q',
                           'cp',
                           path,
                           gs_path))


def create_tb_writer(path, **kwargs):
  alog.debug(f'Creating TB summary writer in {path}')

  return torch.utils.tensorboard.SummaryWriter(path, **kwargs)


def tb_write(tb_writer, name, value, **kwargs):
  if isinstance(value, dict):
    tb_writer.add_scalars(name, value, **kwargs)
  else:
    tb_writer.add_scalar(name, value, **kwargs)


def count_params(net):
  params = 0
  for p in net.parameters():
    params += torch.numel(p)

  return params


def named_grads(net):
  grads = []
  for name, param in net.named_parameters():
    if param.grad is not None:
      grads.append((name, param.grad))
    else:
      alog.debug0(f'Parameter has no gradient: {name}')

  return tuple(grads)


def freeze_params(net, freeze=None, thaw=None):
  freeze = freeze or ()
  thaw = thaw or ()
  for name, param in net.named_parameters():
    requires_grad = None
    for rx in freeze:
      if re.match(rx, name):
        requires_grad = False
        break
    if requires_grad is None:
      for rx in thaw:
        if re.match(rx, name):
          requires_grad = True
          break

    if requires_grad is not None:
      alog.debug(f'{"Thawing" if requires_grad else "Freezing"} parameter "{name}"')
      param.requires_grad = requires_grad
    else:
      alog.debug(f'Parameter "{name}" left untouched (requires_grad={param.requires_grad})')

  return net


def split_dims(shape, npop):
  return shape[: -npop], *shape[-npop:]


def extra_repr(**kwargs):
  rstr = pyu.stri(kwargs)

  return rstr[1: -1]


def kuni_tensor(*shape, dtype=None, device=None, a=None):
  t = torch.empty(*shape, dtype=dtype, device=device)

  nn.init.kaiming_uniform_(t, a=math.sqrt(5) if a is None else a)

  return t


def add(*args):
  return functools.reduce(operator.add, pyu.expand_args(args))


def mul(*args):
  return functools.reduce(operator.mul, pyu.expand_args(args))


def create_graph(x, path=None, params=None, model=None, format='svg'):
  import torchviz

  if params is None and model is not None:
    params = dict(model.named_parameters())

  dot = torchviz.make_dot(x, params=params)

  if path is not None:
    dot.format = format
    dot.render(path)

  return dot


def get_lr(optimizer):
  lrs = []
  for pgrp in optimizer.param_groups:
    lr = pgrp.get('lr')
    if lr is not None:
      lrs.append(lr)

  return np.mean(lrs) if lrs else None


def reset_lr(optimizer, lr):
  for pgrp in optimizer.param_groups:
    if 'lr' in pgrp:
      pgrp['lr'] = lr


def minmax_bbox(bbox):
  # Boxes are (N, 4) shapes with (min_x, min_y, xsize, ysize)
  mmbox = torch.clone(bbox)
  mmbox[:, 2] = bbox[:, 0] + bbox[:, 2]
  mmbox[:, 3] = bbox[:, 1] + bbox[:, 3]

  return mmbox


def get_iou(abox, bbox):
  # Boxes are (..., 4) shapes with (min_x, min_y, max_x, max_y)
  gmask = abox > bbox
  mx = torch.where(gmask, abox, bbox)
  mn = torch.where(gmask, bbox, abox)

  # Intersection box (if not empty) mins are max of mins, and intersection
  # box maxes are min of maxes.
  ixmin = mx[..., 0]
  iymin = mx[..., 1]
  ixmax = mn[..., 2]
  iymax = mn[..., 3]

  idx = ixmax - ixmin
  idx = torch.where(idx > 0, idx, 0)

  idy = iymax - iymin
  idy = torch.where(idy > 0, idy, 0)

  # Intersection areas.
  ia = idx * idy

  # The abox and bbox areas.
  aa = (abox[..., 2] - abox[..., 0]) * (abox[..., 3] - abox[..., 1])
  ab = (bbox[..., 2] - bbox[..., 0]) * (bbox[..., 3] - bbox[..., 1])

  return ia / (aa + ab - ia)


def import_model(modname=None, package=None, path=None, model_args=None):
  name_or_path = path or modname

  module = pymu.import_module(name_or_path, modname=modname, package=package)

  return module.create_model(model_args)

