import collections

import numpy as np
from py_misc_utils import alog
from py_misc_utils import utils as pyu
import torch

from . import utils as ut


STD_PCTILES = (0.0, 0.05, 0.1, 0.5, 0.9, 0.95, 1.0)


def tensor_stats(tensor,
                 name=None,
                 abs_stats=True,
                 percentiles=()):
  if abs_stats:
    tensor = torch.abs(tensor)
  std, mean = torch.std_mean(tensor)
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
  tensor_devices = collections.defaultdict(list)
  stats = []
  for name, tensor in tensor_list:
    stats.append(tensor_stats(tensor,
                              name=name,
                              abs_stats=abs_stats,
                              percentiles=percentiles))
    tensor_devices[tensor.device].append(name)

  device_stats = [f'{prefix} Devices:']
  for dev, names in tensor_devices.items():
    device_stats.append(f'  {dev}\t{names}')

  stats.sort(key=lambda s: getattr(s, sort_by), reverse=True)
  if top_n is not None:
    n = top_n if isinstance(top_n, int) else int(top_n * len(stats))
    stats = stats[: n]

  value_stats = [f'{prefix} Values:']
  for tstat in stats:
    pcts = [f'{int(100 * pp)}%={pv:{fmt}}' for pp, pv in zip(percentiles, tstat.percentile_values)]
    value_stats.append(f'  {tstat.name}\tshape={tstat.shape}\tmin={tstat.min:{fmt}}\t' \
                       f'max={tstat.max:{fmt}}\tmean={tstat.mean:{fmt}}' \
                       f'\tstd={tstat.std:{fmt}}\tpercentiles={pcts}')

  return pyu.make_object(device_stats='\n'.join(device_stats),
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


def show_tensors_stats(tstats, slevs):
  for k, v in slevs.items():
    alog.log(v, getattr(tstats, k))

