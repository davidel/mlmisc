import collections

from py_misc_utils import alog
from py_misc_utils import utils as pyu
import torch


_STD_PCTILES = (0.0, 0.05, 0.1, 0.5, 0.9, 0.95, 1.0)


def tensor_stats(tensor,
                 name=None,
                 abs_stats=True,
                 percentiles=_STD_PCTILES):
  if abs_stats:
    tensor = torch.abs(tensor)
  pct_tensor = torch.tensor(percentiles).to(tensor.device)
  std, mean = torch.std_mean(tensor)
  quantile = torch.quantile(tensor, pct_tensor)

  return pyu.make_object(name=name,
                         shape=tuple(tensor.shape),
                         std=std.item(),
                         mean=mean.item(),
                         percentile_values=quantile.tolist(),
                         percentiles=percentiles)


def get_tensors_stats(prefix, tensor_list,
                      abs_stats=True,
                      percentiles=_STD_PCTILES):
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

  stats.sort(key=lambda s: s.mean)

  value_stats = [f'{prefix}: percentiles={percentiles}']
  for tstat in stats:
    pcts = pyu.format(tstat.percentile_values, '.5e')
    value_stats.append(f'  {tstat.name}\tshape={tstat.shape}\tmean={tstat.mean:.5e}' \
                       f'\tstd={tstat.std:.5e}\tpercentiles={pcts}')

  pct_stats = [f'{prefix} Percentiles:']
  for gp in sorted(percentiles):
    x = min(int(gp * len(stats)), len(stats) - 1)
    tstat = stats[x]
    pct_stats.append(f'  {gp * 100:.1f}%\t= {tstat.mean:.5e} ({tstat.name})')

  return pyu.make_object(device_stats='\n'.join(device_stats),
                         value_stats='\n'.join(value_stats),
                         pct_stats='\n'.join(pct_stats))


def get_parameters_stats(model, percentiles=_STD_PCTILES):
  return get_tensors_stats('Parameters', model.named_parameters(),
                           percentiles=percentiles)


def get_grads_stats(model, percentiles=_STD_PCTILES):
  grads = []
  for name, param in model.named_parameters():
    if param.grad is not None:
      grads.append((name, param.grad))
    else:
      alog.debug0(f'Parameter has no gradient: {name}')

  return get_tensors_stats('Gradients', grads, percentiles=percentiles)


def show_tensors_stats(tstats, slevs):
  for k, v in slevs.items():
    alog.log(v, getattr(tstats, k))

