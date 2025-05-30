import os
import pickle

import numpy as np
import py_misc_utils.alog as alog
import py_misc_utils.assert_checks as tas
import py_misc_utils.core_utils as pycu
import py_misc_utils.file_overwrite as pyfow
import py_misc_utils.gfs as gfs
import py_misc_utils.inspect_utils as pyiu
import py_misc_utils.pickle_wrap as pypw
import py_misc_utils.utils as pyu
import torch

from . import auto_module as am
from . import core_utils as cu
from . import load_state_dict as lsd


pypw.add_known_module(__name__)


def model_save(model, path):
  alog.debug(f'Saving model to {path} ...')
  with pyfow.FileOverwrite(path, mode='wb') as ptfd:
    state = model.state_dict()
    cu.nan_check(state, name='STATE')
    torch.save(state, ptfd)
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


def checkpoint_model(model, path, rmt_path=None):
  model_save(model, path)

  if rmt_path is not None:
    alog.debug(f'Copying model state to {rmt_path}')
    gfs.copy(path, rmt_path)


def save_data(path, **kwargs):
  data = dict()
  for name, ndata in kwargs.items():
    sdfn = getattr(ndata, 'state_dict', None)
    if sdfn is not None and callable(sdfn):
      state = sdfn()
      cu.nan_check(state, name='STATE')
      data[name] = state
    else:
      data[name] = pypw.wrap(ndata)

  alog.debug(f'Saving data to {path} ...')
  with pyfow.FileOverwrite(path, mode='wb') as ptfd:
    torch.save(data, ptfd)
  alog.debug(f'Data saved to {path}')


def load_state(torch_data, strict=None, **kwargs):
  data = dict()
  for name, tdata in torch_data.items():
    xdata = pypw.unwrap(tdata)

    sdobj = kwargs.get(name)
    if sdobj is not None:
      lsd.load_state_dict(sdobj, xdata, strict=strict)
      data[name] = sdobj
    elif pycu.isdict(xdata) and am.is_auto_state(xdata):
      data[name] = am.load(xdata, strict=strict)
    else:
      data[name] = xdata

  return data


def load_data(path, map_location=None, strict=None, **kwargs):
  alog.debug(f'Loading data from {path}')
  torch_data = cu.torch_load(path, map_location=map_location or torch.device('cpu'))

  return load_state(torch_data, strict=strict, **kwargs)


def load_raw_data(path, map_location=None):
  alog.debug(f'Loading data from {path}')
  torch_data = cu.torch_load(path, map_location=map_location or torch.device('cpu'))

  return pypw.unwrap(torch_data)


def checkpoint_data(path, rmt_path=None, **kwargs):
  save_data(path, **kwargs)

  if rmt_path is not None:
    alog.debug(f'Copying data to {rmt_path}')
    gfs.copy(path, rmt_path)

