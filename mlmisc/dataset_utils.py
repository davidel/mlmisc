import inspect
import os
import random

import datasets as dsets
import huggingface_hub as hfh
import py_misc_utils.alog as alog
import torch
import torchvision

from . import utils as ut


def _create_torchvision_dataset(dsclass, **kwargs):
  sig = inspect.signature(dsclass)
  train = kwargs.get('train')
  if train is not None and sig.parameters.get('train') is None:
    kwargs.pop('train')
    kwargs['split'] = 'train' if train else 'test'

  download = kwargs.get('download')
  if download is not None and sig.parameters.get('download') is None:
    alog.info(f'Dropping download={download} argument')
    kwargs.pop('download')

  return dsclass(**kwargs)


def _try_torchvision(name,root, transform, target_transform):
  dsclass = getattr(torchvision.datasets, name, None)
  if dsclass is not None:
    ds = dict()
    ds['train'] = _create_torchvision_dataset(dsclass,
                                              root=root,
                                              train=True,
                                              transform=transform.get('train'),
                                              target_transform=target_transform.get('train'),
                                              download=True)
    ds['test'] = _create_torchvision_dataset(dsclass,
                                             root=root,
                                             train=False,
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


def dict_selector(keys):

  def select_fn(x):
    return [x[k] for k in keys]

  return select_fn


class HFDataset(torch.utils.data.Dataset):

  def __init__(self, data,
               select_fn=None,
               transform=None,
               target_transform=None):
    super().__init__()
    self.data = data
    self.select_fn = select_fn or _guess_select
    self.transform = transform or _no_transform
    self.target_transform = target_transform or _no_transform

  def __len__(self):
    return len(self.data)

  def __getitem__(self, i):
    x, y = self.select_fn(self.data[i])

    return self.transform(x), self.target_transform(y)


def _norm_transforms(transform):
  if isinstance(transform, dict):
    return transform

  return dict(train=transform, test=transform)


def create_dataset(name,
                   root=None,
                   select_fn=None,
                   transform=None,
                   target_transform=None):
  root = root or os.path.join(os.getenv('HOME', '.'), 'datasets')
  transform = _norm_transforms(transform)
  target_transform = _norm_transforms(target_transform)

  if name.find('/') < 0:
    ds = _try_torchvision(name,
                          root=root,
                          transform=transform,
                          target_transform=target_transform)
    if ds is not None:
      return ds

  if hfh.list_datasets(dataset_name=name):
    hfds = dsets.load_dataset(name, cache_dir=root)

    ds = dict()
    ds['train'] = HFDataset(hfds['train'],
                            select_fn=select_fn,
                            transform=transform.get('train'),
                            target_transform=target_transform.get('train'))
    ds['test'] = HFDataset(hfds['test'],
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

