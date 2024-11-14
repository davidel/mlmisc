import array
import random

import py_misc_utils.alog as alog
import py_misc_utils.assert_checks as tas
import py_misc_utils.utils as pyu
import torch

from . import utils as ut


class DatasetBase:

  def __init__(self, select_fn=None, transform=None, target_transform=None, **kwargs):
    self._select_fn = pyu.value_or(select_fn, ident_select)
    self._transform = pyu.value_or(transform, no_transform)
    self._target_transform = pyu.value_or(target_transform, no_transform)
    self._kwargs = kwargs

  def extra_arg(self, name):
    return self._kwargs.get(name)

  def process_sample(self, data):
    x, y = self._select_fn(data)

    return self._transform(x), self._target_transform(y)


class Dataset(torch.utils.data.Dataset, DatasetBase):

  def __init__(self, select_fn=None, transform=None, target_transform=None, **kwargs):
    torch.utils.data.Dataset.__init__(self)
    DatasetBase.__init__(self,
                         select_fn=select_fn,
                         transform=transform,
                         target_transform=target_transform,
                         **kwargs)

  def __getitem__(self, i):
    if isinstance(i, slice):
      return sliced_dataset(self, i)

    data = self.get_sample(i)

    return self.process_sample(data)


class IterableDataset(torch.utils.data.IterableDataset, DatasetBase):

  def __init__(self, select_fn=None, transform=None, target_transform=None, **kwargs):
    torch.utils.data.IterableDataset.__init__(self)
    DatasetBase.__init__(self,
                         select_fn=select_fn,
                         transform=transform,
                         target_transform=target_transform,
                         **kwargs)

  def generate(self):
    for data in self.enum_samples():
      yield self.process_sample(data)

  def __iter__(self):
    return iter(self.generate())

  def __len__(self):
    return self.extra_arg('size')


class SubDataset(torch.utils.data.Dataset):

  def __init__(self, data, indices):
    super().__init__()
    self._data = data
    self._indices = indices

  def extra_arg(self, name):
    extra_arg = getattr(self._data, 'extra_arg', None)

    return extra_arg(name) if extra_arg is not None else getattr(self._data, name, None)

  def __len__(self):
    return len(self._indices)

  def __getitem__(self, i):
    if isinstance(i, slice):
      return SubDataset(self._data, self._indices[i])

    return self._data[self._indices[i]]


def no_transform(x):
  return x


def to_transform(**kwargs):
  def transform(x):
    return x.to(**kwargs)

  return transform


def ident_select(x):
  return x


def guess_select(x):
  if isinstance(x, (list, tuple)):
    return x[: 2]
  if isinstance(x, dict):
    return tuple(x.values())[: 2]

  return x


def sliced_dataset(ds, dslice):
  ds_size = len(ds)
  indices = array.array(pyu.array_code(ds_size), range(*dslice.indices(ds_size)))

  return SubDataset(ds, indices)


_DS_SEED = pyu.getenv('DS_SEED', dtype=int, defval=997727)

def shuffled_indices(size, seed=None):
  seed = pyu.value_or(seed, _DS_SEED)

  rng = random.Random(seed)
  indices = array.array(pyu.array_code(size), range(size))
  rng.shuffle(indices)

  return indices

