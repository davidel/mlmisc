import array
import random

import py_misc_utils.alog as alog
import py_misc_utils.assert_checks as tas
import py_misc_utils.utils as pyu
import torch

from . import utils as ut


class Pipeline:

  def __init__(self, init=None):
    self._elems = list(init) if init else []

  def __len__(self):
    return len(self._elems)

  def add(self, elem):
    self._elems.append(elem)

  def __call__(self, x):
    for elem in self._elems:
      x = elem(x)

    return x


class DatasetBase:

  def __init__(self, pipeline=None, **kwargs):
    self._pipeline = pyu.value_or(pipeline, lambda x: x)
    self._kwargs = kwargs

  def extra_arg(self, name):
    return self._kwargs.get(name)

  def process_sample(self, data):
    return self._pipeline(data)


class Dataset(torch.utils.data.Dataset, DatasetBase):

  def __init__(self, pipeline=None, **kwargs):
    torch.utils.data.Dataset.__init__(self)
    DatasetBase.__init__(self, pipeline=pipeline, **kwargs)

  def __getitem__(self, i):
    if isinstance(i, slice):
      return sliced_dataset(self, i)

    data = self.get_sample(i)

    return self.process_sample(data)


class IterableDataset(torch.utils.data.IterableDataset, DatasetBase):

  def __init__(self, pipeline=None, **kwargs):
    torch.utils.data.IterableDataset.__init__(self)
    DatasetBase.__init__(self, pipeline=pipeline, **kwargs)

  def generate(self):
    for data in self.enum_samples():
      yield self.process_sample(data)

  def __iter__(self):
    return iter(self.generate())

  def __len__(self):
    return self.extra_arg('size')


class ShufflerDataset(torch.utils.data.IterableDataset):

  def __init__(self, data, buffer_size=None):
    buffer_size = pyu.value_or(buffer_size, 1000)

    super().__init__()
    self._data = data
    self._buffer_size = buffer_size

  def generate(self):
    stacked = []
    for data in self._data:
      if self._buffer_size > len(stacked):
        stacked.append(data)
      else:
        idx = random.randrange(self._buffer_size)
        yield stacked[idx]
        stacked[idx] = data

    random.shuffle(stacked)
    for data in stacked:
      yield data

  def __iter__(self):
    return iter(self.generate())

  def __len__(self):
    return len(self._data)


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


def to_transform(**kwargs):
  def transform(x):
    return x.to(**kwargs)

  return transform


def transformer(sample=None, target=None):
  sample = sample or (lambda x: x)
  target = target or (lambda x: x)

  def transformer_fn(x):
    x, y = x

    return sample(x), target(y)

  return transformer_fn


def items_selector(items):

  def select_fn(x):
    return [x[i] for i in items]

  return select_fn


def guess_select():

  def select_fn(x):
    if isinstance(x, (list, tuple)):
      return x[: 2]
    if isinstance(x, dict):
      return tuple(x.values())[: 2]

    return x

  return select_fn


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


def shuffled_data(data, seed=None):
  indices = shuffled_indices(len(data), seed=seed)

  return type(data)(data[i] for i in indices)

