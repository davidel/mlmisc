import collections

import numpy as np
import py_misc_utils.alog as alog
import py_misc_utils.core_utils as pycu
import py_misc_utils.object_tracker as pyot
import py_misc_utils.utils as pyu
import torch
import torch.nn as nn

from . import core_utils as cu


STD_PCTILES = (0.0, 0.05, 0.1, 0.5, 0.9, 0.95, 1.0)


def tensor_stats(tensor,
                 name=None,
                 abs_stats=True,
                 percentiles=()):
  if abs_stats:
    tensor = torch.abs(tensor)
  # Avoid size==1 tensors to trigger warnings within the torch.std_mean() API.
  std, mean = torch.std_mean(tensor, correction=min(1, torch.numel(tensor) - 1))
  mm = torch.aminmax(tensor)

  if percentiles:
    # Run the quantile with Numpy as Pytorch has too strict tensor size limits.
    quantile = np.quantile(tensor.numpy(force=True), percentiles).tolist()
  else:
    quantile = []

  return pyu.make_object(name=name,
                         shape=tuple(tensor.shape),
                         min=mm.min.item(),
                         max=mm.max.item(),
                         std=std.item(),
                         mean=mean.item(),
                         percentile_values=quantile,
                         percentiles=percentiles)


def get_tensors_stats(prefix, tensor_list,
                      abs_stats=True,
                      sort_by='mean',
                      percentiles=(),
                      top_n=None,
                      fmt='.3e'):
  stats = [tensor_stats(tensor,
                        name=name,
                        abs_stats=abs_stats,
                        percentiles=percentiles)
           for name, tensor in tensor_list]

  stats.sort(key=lambda s: getattr(s, sort_by), reverse=True)
  if top_n is not None:
    n = top_n if isinstance(top_n, int) else int(top_n * len(stats))
    stats = stats[: n]

  value_stats = [f'{prefix} Values:']
  for ts in stats:
    pcts = ', '.join(f'{int(100 * pp)}%={pv:{fmt}}'
                    for pp, pv in zip(percentiles, ts.percentile_values))
    stat_line = ' '.join([f' min={ts.min:{fmt}}',
                          f' max={ts.max:{fmt}}',
                          f'mean={ts.mean:{fmt}}',
                          f' std={ts.std:{fmt}}',
                          f'pcts=({pcts})'])

    value_stats.append(f'  {ts.name} {ts.shape}:')
    value_stats.append(f'    {stat_line}')

  return pyu.make_object(stats=stats,
                         value_stats='\n'.join(value_stats))


def get_parameters_stats(model, **kwargs):
  return get_tensors_stats('Parameters', model.named_parameters(), **kwargs)


def get_grads_stats(model, **kwargs):
  return get_tensors_stats('Gradients', cu.named_grads(model), **kwargs)


def show_tensors_stats(stats, slevs):
  for k, v in slevs.items():
    alog.log(v, getattr(stats, k))


class TensorTracker:

  def __init__(self, min_size=None):
    self._min_size = min_size

  def track(self, obj):
    size = info = None
    if torch.is_tensor(obj):
      size = obj.element_size() * obj.nelement()
      info = f'PT Tensor: shape={tuple(obj.shape)} dtype={obj.dtype} device={obj.device} ' \
        f'size={pycu.size_str(size)}'
    elif isinstance(obj, np.ndarray):
      size = obj.size * obj.itemsize
      info = f'NP Tensor: shape={tuple(obj.shape)} dtype={obj.dtype} size={pycu.size_str(size)}'

    if info is not None and (self._min_size is None or size >= self._min_size):
      return size, info


def track_tensors(min_size=None, max_references=None):
  tracker = TensorTracker(min_size=min_size)

  return pyot.track_objects(tracker, max_references=max_references)


def _nan_hook(module, x, y):
  try:
    cu.nan_check(x, name='X')
    cu.nan_check(y, name='Y')
  except ValueError as ex:
    ex.add_note(f'In module: {module}')
    raise ex


_HOOK_HANDLE = None

def set_detect_anomaly(activate):
  global _HOOK_HANDLE

  if activate:
    if _HOOK_HANDLE is None:
      _HOOK_HANDLE = nn.modules.module.register_module_forward_hook(_nan_hook)
  else:
    _HOOK_HANDLE.remove()
    _HOOK_HANDLE = None

  torch.autograd.set_detect_anomaly(activate)

