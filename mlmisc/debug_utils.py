import collections

from py_misc_utils import alog
from py_misc_utils import utils as pyu
import torch


STD_PCTILES = (0.0, 0.05, 0.1, 0.5, 0.9, 0.95, 1.0)


def tensor_stats(tensor,
                 name=None,
                 device=None,
                 abs_stats=True,
                 percentiles=()):
  if device is not None:
    tensor = tensor.to(device)
  if abs_stats:
    tensor = torch.abs(tensor)
  std, mean = torch.std_mean(tensor)

  if percentiles:
    pct_tensor = torch.tensor(percentiles).to(tensor.device)
    quantile = torch.quantile(tensor, pct_tensor).tolist()
  else:
    quantile = []

  return pyu.make_object(name=name,
                         shape=tuple(tensor.shape),
                         min=torch.min(tensor).item(),
                         max=torch.max(tensor).item(),
                         std=std.item(),
                         mean=mean.item(),
                         percentile_values=quantile,
                         percentiles=percentiles)


def get_tensors_stats(prefix, tensor_list,
                      device=None,
                      abs_stats=True,
                      sort_by='mean',
                      percentiles=(),
                      top_n=None):
  tensor_devices = collections.defaultdict(list)
  stats = []
  for name, tensor in tensor_list:
    stats.append(tensor_stats(tensor,
                              name=name,
                              device=device,
                              abs_stats=abs_stats,
                              percentiles=percentiles))
    tensor_devices[tensor.device].append(name)

  device_stats = [f'{prefix} Devices:']
  for dev, names in tensor_devices.items():
    device_stats.append(f'  {dev}\t{names}')

  stats.sort(key=lambda s: getattr(s, sort_by))
  if top_n is not None:
    n = top_n if isinstance(top_n, int) else int(top_n * len(stats))
    stats = stats[-n:]

  value_stats = [f'{prefix} Values:']
  for tstat in stats:
    pcts = pyu.format(tstat.percentile_values, '.5e')
    value_stats.append(f'  {tstat.name}\tshape={tstat.shape}\tmin={tstat.min:.5e}\t' \
                       f'max={tstat.max:.5e}\tmean={tstat.mean:.5e}' \
                       f'\tstd={tstat.std:.5e}\tpercentiles={pcts}')

  return pyu.make_object(device_stats='\n'.join(device_stats),
                         value_stats='\n'.join(value_stats))


def get_parameters_stats(model,
                         device=None,
                         abs_stats=True,
                         sort_by='mean',
                         percentiles=(),
                         top_n=None):
  return get_tensors_stats('Parameters', model.named_parameters(),
                           device=device,
                           abs_stats=abs_stats,
                           sort_by=sort_by,
                           percentiles=percentiles,
                           top_n=top_n)


def get_grads_stats(model,
                    device=None,
                    abs_stats=True,
                    sort_by='mean',
                    percentiles=(),
                    top_n=None):
  grads = []
  for name, param in model.named_parameters():
    if param.grad is not None:
      grads.append((name, param.grad))
    else:
      alog.debug0(f'Parameter has no gradient: {name}')

  return get_tensors_stats('Gradients', grads,
                           device=device,
                           abs_stats=abs_stats,
                           sort_by=sort_by,
                           percentiles=percentiles,
                           top_n=top_n)


def show_tensors_stats(tstats, slevs):
  for k, v in slevs.items():
    alog.log(v, getattr(tstats, k))

