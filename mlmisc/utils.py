import importlib
import os
import random
import subprocess
import sys

import numpy as np
import py_misc_utils.alog as alog
import torch

from . import auto_module as am


def get_device(kind=None):
  if kind is None:
    kind = 'cuda' if torch.cuda.is_available() else 'cpu'

  return torch.device(kind)


def model_shape(model, shape, device=None):
  model.eval()
  with torch.no_grad():
    zin = torch.zeros((1,) + tuple(shape), device=device)
    out = model(zin)

    return out.shape[1:]


def model_save(model, path):
  alog.debug(f'Saving model to {path} ...')
  torch.save(model.state_dict(), path)
  alog.debug(f'Model saved to {path}')


def model_load(model, path, device=None):
  if os.path.exists(path):
    alog.debug(f'Loading model state from {path}')
    model.load_state_dict(torch.load(path))

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
  torch.save(data, path)
  alog.debug(f'Data saved to {path}')


def load_data(path, **kwargs):
  alog.debug(f'Loading data from {path}')
  td = torch.load(path)

  data = dict()
  for name, ndata in td.items():
    sdobj = kwargs.get(name, None)
    if sdobj is not None:
      sdobj.load_state_dict(ndata)
    elif isinstance(ndata, dict) and am.is_module(ndata):
      data[name] = am.load_module(ndata)
    else
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


def get_lr(optimizer):
  lrs = []
  for group in optimizer.param_groups:
    lr = group.get('lr', None)
    if lr is not None:
      lrs.append(lr)

  return sum(lrs) / len(lrs) if lrs else 0.0


def reset_lr(optimizer, lr):
  for g in optimizer.param_groups:
    g['lr'] = lr


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

