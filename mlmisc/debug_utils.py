import collections

import numpy as np
import py_misc_utils.alog as alog
import py_misc_utils.core_utils as pycu
import py_misc_utils.object_tracker as pyot
import py_misc_utils.utils as pyu
import torch

from . import utils as ut


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
    pcts = [f'{int(100 * pp)}%={pv:{fmt}}' for pp, pv in zip(percentiles, ts.percentile_values)]
    value_stats.append(f'  {ts.name}\tshape={ts.shape}\tmin={ts.min:{fmt}}\t' \
                       f'max={ts.max:{fmt}}\tmean={ts.mean:{fmt}}' \
                       f'\tstd={ts.std:{fmt}}\tpercentiles={pcts}')

  return pyu.make_object(stats=stats,
                         value_stats='\n'.join(value_stats))


def get_parameters_stats(model,
                         abs_stats=True,
                         sort_by='mean',
                         percentiles=(),
                         top_n=None):
  return get_tensors_stats('Parameters', model.named_parameters(),
                           abs_stats=abs_stats,
                           sort_by=sort_by,
                           percentiles=percentiles,
                           top_n=top_n)


def get_grads_stats(model,
                    abs_stats=True,
                    sort_by='mean',
                    percentiles=(),
                    top_n=None):
  return get_tensors_stats('Gradients', ut.named_grads(model),
                           abs_stats=abs_stats,
                           sort_by=sort_by,
                           percentiles=percentiles,
                           top_n=top_n)


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


def track_tensors(min_size=None):
  tracker = TensorTracker(min_size=min_size)

  return pyot.track_objects(tracker)

