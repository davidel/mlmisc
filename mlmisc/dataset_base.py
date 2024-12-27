import array
import functools
import random

import py_misc_utils.alog as alog
import py_misc_utils.assert_checks as tas
import py_misc_utils.core_utils as pycu
import py_misc_utils.utils as pyu
import torch

from . import utils as ut


class DatasetBase:

  def __init__(self, pipeline=None, **kwargs):
    self._pipeline = pipeline
    self._kwargs = kwargs

  def extra_arg(self, name):
    return self._kwargs.get(name)

  def process_sample(self, data):
    return self._pipeline(data) if self._pipeline is not None else data

  def reset_pipeline(self, new_pipeline=None):
    pipeline, self._pipeline = self._pipeline, new_pipeline

    return pipeline


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
    buffer_size = pyu.value_or(buffer_size, 1024)

    super().__init__()
    self._data = data
    self._buffer_size = buffer_size

  def generate(self):
    stacked = []
    for data in self._data:
      if self._buffer_size > len(stacked):
        stacked.append(data)
      else:
        idx = random.randrange(len(stacked))
        yield stacked[idx]
        stacked[idx] = data

    random.shuffle(stacked)
    for data in stacked:
      yield data

  def __iter__(self):
    return iter(self.generate())

  def __len__(self):
    data_length = getattr(self._data, '__len__', None)

    return data_length() if data_length is not None else None


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


def _to_transform_fn(x, **kwargs):
  return x.to(**kwargs)


def to_transform(**kwargs):
  return functools.partial(_to_transform_fn, **kwargs)


def _transformer_fn(sample, target, x):
  s, t = x

  return sample(s), target(t)


def transformer(sample=None, target=None):
  sample = pyu.value_or(sample, pycu.ident)
  target = pyu.value_or(target, pycu.ident)

  return functools.partial(_transformer_fn, sample, target)


def _items_selector_fn(items, x):
  return [x[i] for i in items]


def items_selector(items):
  return functools.partial(_items_selector_fn, items)


def sliced_dataset(dataset, dslice):
  indices = array.array(pyu.array_code(len(dataset)),
                        range(*dslice.indices(len(dataset))))

  return SubDataset(dataset, indices)


_DATASET_SEED = pyu.getenv('DATASET_SEED', dtype=int, defval=997727)

def shuffled_indices(size, seed=None):
  seed = pyu.value_or(seed, _DATASET_SEED)

  rng = random.Random(seed)
  indices = array.array(pyu.array_code(size), range(size))
  rng.shuffle(indices)

  return indices


def shuffled_data(data, seed=None):
  indices = shuffled_indices(len(data), seed=seed)

  return type(data)(data[i] for i in indices)

