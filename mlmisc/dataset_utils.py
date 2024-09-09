import inspect
import os
import random

import datasets as dsets
import huggingface_hub as hfh
import py_misc_utils.alog as alog
import py_misc_utils.utils as pyu
import torch
import torchvision

from . import utils as ut


class Dataset(torch.utils.data.Dataset):

  def __init__(self, data,
               select_fn=None,
               transform=None,
               target_transform=None):
    super().__init__()
    self.data = data
    self.select_fn = select_fn or _guess_select
    self.transform = transform or _no_transform
    self.target_transform = target_transform or _no_transform

  def _get(self, i):
    if isinstance(self.data, dict):
      return {k: v[i] for k, v in self.data.items()}

    return self.data[i]

  def __len__(self):
    if isinstance(self.data, dict):
      return min(len(v) for v in self.data.values())

    return len(self.data)

  def __getitem__(self, i):
    idata = self._get(i)

    if isinstance(i, slice):
      return Dataset(idata,
                     select_fn=self.select_fn,
                     transform=self.transform,
                     target_transform=self.target_transform)

    x, y = self.select_fn(idata)

    return self.transform(x), self.target_transform(y)


def _try_torchvision(name, root, transform, target_transform, split_pct):
  dsclass = getattr(torchvision.datasets, name, None)
  if dsclass is not None:
    sig = inspect.signature(dsclass)
    kwargs = dict(download=True) if sig.parameters.get('download') is not None else dict()

    ds = dict()
    if sig.parameters.get('train') is not None:
      ds['train'] = dsclass(root=root,
                            train=True,
                            transform=transform.get('train'),
                            target_transform=target_transform.get('train'),
                            **kwargs)
      ds['test'] = dsclass(root=root,
                           train=False,
                           transform=transform.get('test'),
                           target_transform=target_transform.get('test'),
                           **kwargs)
    elif sig.parameters.get('split') is not None:
      ds['train'] = dsclass(root=root,
                            split='train',
                            transform=transform.get('train'),
                            target_transform=target_transform.get('train'),
                            **kwargs)
      ds['test'] = dsclass(root=root,
                           split='test',
                           transform=transform.get('test'),
                           target_transform=target_transform.get('test'),
                           **kwargs)
    else:
      full_ds = dsclass(root=root, **kwargs)

      ntrain = int(split_pct * len(full_ds))

      ds['train'] = Dataset(full_ds[: ntrain],
                            transform=transform.get('train'),
                            target_transform=target_transform.get('train'))
      ds['test'] = Dataset(full_ds[ntrain: ],
                           transform=transform.get('test'),
                           target_transform=target_transform.get('test'))

    return ds


def _guess_select(x):
  if isinstance(x, (list, tuple)):
    return x[: 2]
  if isinstance(x, dict):
    return list(x.values())[: 2]

  return x


def _no_transform(x):
  return x


def keys_selector(keys):

  def select_fn(x):
    return [x[k] for k in keys]

  return select_fn


def _norm_transforms(transform):
  if isinstance(transform, dict):
    return transform

  return dict(train=transform, test=transform)


def create_dataset(name,
                   root=None,
                   select_fn=None,
                   transform=None,
                   target_transform=None,
                   split_pct=None):
  root = root or os.path.join(os.getenv('HOME', '.'), 'datasets')
  transform = _norm_transforms(transform)
  target_transform = _norm_transforms(target_transform)
  split_pct = split_pct or 0.9

  if name.find('/') < 0:
    ds = _try_torchvision(name, root, transform, target_transform, split_pct)
    if ds is not None:
      return ds

  if hfh.list_datasets(dataset_name=name):
    hfds = dsets.load_dataset(name, cache_dir=root)

    ds = dict()
    ds['train'] = Dataset(hfds['train'],
                          select_fn=select_fn,
                          transform=transform.get('train'),
                          target_transform=target_transform.get('train'))
    ds['test'] = Dataset(hfds['test'],
                         select_fn=select_fn,
                         transform=transform.get('test'),
                         target_transform=target_transform.get('test'))

    return ds

  alog.xraise(ValueError, f'Unable to create dataset: "{name}"')


def get_class_weights(data,
                      dtype=None,
                      cdtype=None,
                      output_filter=None,
                      max_samples=None):
  # By default assume target is the second entry in the dataset return tuple.
  output_filter = output_filter or (lambda x: x[1])

  indices = list(range(len(data)))
  if max_samples is not None and len(indices) > max_samples:
    random.shuffle(indices)
    indices = sorted(indices[: max_samples])

  target = torch.empty(len(indices), dtype=cdtype or torch.int32)
  for i in indices:
    y = output_filter(data[i])
    target[i] = ut.item(y)

  cvalues, class_counts = torch.unique(target, return_counts=True)
  weight = 1.0 / class_counts
  weight = weight / torch.sum(weight)

  if dtype is not None:
    weight = weight.to(dtype)

  if ut.is_integer(cvalues):
    max_class = torch.max(cvalues).item()
    if max_class >= len(cvalues):
      fweight = torch.zeros(max_class + 1, dtype=weight.dtype)
      fweight[cvalues] = weight
      weight = fweight

  alog.debug(f'Data class weight: { {c: f"{n:.2e}" for c, n in enumerate(weight)} }')

  return weight

