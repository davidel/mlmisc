import importlib
import os
import random
import subprocess
import sys

import numpy as np
import py_misc_utils.alog as alog
import py_misc_utils.file_overwrite as pyfow
import py_misc_utils.utils as pyu
import torch
import torch.utils.tensorboard

from . import auto_module as am


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


def torch_load(path, **kwargs):
  return torch.load(path, weights_only=False, **kwargs)


def net_shape(net, shape, device=None, output_select=None):
  with torch.no_grad(), Training(net, False):
    zin = torch.zeros((1,) + tuple(shape), device=device)
    out = net(zin)
    out = out if output_select is None else output_select(out)

    return out.shape[1:]


def model_save(model, path):
  alog.debug(f'Saving model to {path} ...')
  with pyfow.FileOverwrite(path, mode='wb') as ptfd:
    torch.save(model.state_dict(), ptfd)
  alog.debug(f'Model saved to {path}')


def model_load(path, model=None, device=None, strict=True):
  map_location = torch.device('cpu') if device is not None else None
  if model is None:
    alog.debug(f'Loading model state from {path}')
    model = am.load(path, map_location=map_location, strict=strict)
  elif os.path.exists(path):
    alog.debug(f'Loading model state from {path}')
    model.load_state_dict(torch.load(path, map_location=map_location),
                          strict=strict, weights_only=True)

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


def load_data(path, map_location=None, strict=True, **kwargs):
  alog.debug(f'Loading data from {path}')
  td = torch_load(path, map_location=map_location or torch.device('cpu'))

  data = dict()
  for name, ndata in td.items():
    sdobj = kwargs.get(name, None)
    if sdobj is not None:
      sdobj.load_state_dict(ndata, strict=strict)
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


def create_tb_writer(path):
  alog.debug(f'Creating TB summary writer in {path}')

  return torch.utils.tensorboard.SummaryWriter(path)


def count_params(net):
  params = 0
  for p in net.parameters():
    params += torch.numel(p)

  return params


def tail_permute(t):
  perm = list(range(t.ndim))
  perm[-1], perm[-2] = perm[-2], perm[-1]

  return torch.permute(t, tuple(perm))


def split_dims(shape, npop):
  return shape[: -npop], *shape[-npop: ]


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


def import_model(module_name=None, package=None, path=None, model_args=None):
  if path is not None:
    if module_name is None:
      module_name = os.path.basename(path).split('.', 1)[0]

    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
  else:
    mod = importlib.import_module(module_name, package=package)

  return mod.create_model(model_args)

