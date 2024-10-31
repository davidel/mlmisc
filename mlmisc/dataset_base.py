import array
import random

import py_misc_utils.alog as alog
import py_misc_utils.assert_checks as tas
import py_misc_utils.utils as pyu
import torch

from . import utils as ut


class Dataset(torch.utils.data.Dataset):

  def __init__(self, select_fn=None, transform=None, target_transform=None, **kwargs):
    super().__init__()
    self.select_fn = select_fn or ident_select
    self.transform = transform or no_transform
    self.target_transform = target_transform or no_transform
    self.kwargs = kwargs

  def extra_arg(self, name):
    return self.kwargs.get(name)

  def __getitem__(self, i):
    if isinstance(i, slice):
      return sliced_dataset(self, i)

    data = self.get_sample(i)
    x, y = self.select_fn(data)

    return self.transform(x), self.target_transform(y)


class SubDataset(torch.utils.data.Dataset):

  def __init__(self, data, indices):
    super().__init__()
    self.data = data
    self.indices = indices

  def extra_arg(self, name):
    extra_arg = getattr(self.data, 'extra_arg', None)

    return extra_arg(name) if extra_arg is not None else getattr(self.data, name, None)

  def __len__(self):
    return len(self.indices)

  def __getitem__(self, i):
    if isinstance(i, slice):
      return SubDataset(self.data, self.indices[i])

    return self.data[self.indices[i]]


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


_DS_SEED = pyu.getenv('DS_SEED', dtype=int, defval=9041934)

def shuffled_indices(size, seed=None):
  seed = pyu.value_or(seed, _DS_SEED)

  rng = random.Random(seed)
  indices = array.array(pyu.array_code(size), range(size))
  rng.shuffle(indices)

  return indices

