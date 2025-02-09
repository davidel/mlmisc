import array
import functools
import random

import py_misc_utils.alog as alog
import py_misc_utils.core_utils as pycu
import py_misc_utils.pipeline as pypl
import py_misc_utils.utils as pyu
import torch


class DatasetBase:

  def __init__(self, pipeline=None, **kwargs):
    self._pipeline = pyu.value_or(pipeline, pypl.Pipeline())
    self._kwargs = kwargs

  def extra_arg(self, name):
    return self._kwargs.get(name)

  def add_extra_arg(self, name, value):
    self._kwargs[name] = value

  def process_sample(self, data):
    return self._pipeline(data)

  def pipeline(self):
    return self._pipeline.clone()


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


class DatasetWrapper:

  def __init__(self, data, **kwargs):
    self._data = data
    self._kwargs = kwargs

  def extra_arg(self, name):
    value = self._kwargs.get(name)
    if value is not None:
      return value

    for source in (super(), self._data):
      extra_arg = getattr(source, 'extra_arg', None)
      if extra_arg is not None and (value := extra_arg(name)) is not None:
        return value

    return getattr(self._data, name, None)

  def add_extra_arg(self, name, value):
    self._kwargs[name] = value

  def __len__(self):
    for source in (super(), self._data):
      data_length = getattr(source, '__len__', None)
      if data_length is not None:
        return data_length()

    return self.extra_arg('size')


class SubDataset(torch.utils.data.Dataset, DatasetWrapper):

  def __init__(self, data, indices):
    torch.utils.data.Dataset.__init__(self)
    DatasetWrapper.__init__(self, data)
    self._indices = indices

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


def transformer(sample=pycu.ident, target=pycu.ident):
  return functools.partial(_transformer_fn, sample, target)


def _items_selector_fn(items, x):
  return [x[i] for i in items] if isinstance(items, (list, tuple)) else x[items]


def items_selector(items):
  return functools.partial(_items_selector_fn, items)


def sliced_dataset(dataset, dslice):
  indices = array.array(pyu.array_code(len(dataset)),
                        range(*dslice.indices(len(dataset))))

  return SubDataset(dataset, indices)


_DATASET_SEED = pyu.getenv('DATASET_SEED', dtype=int, defval=997727)

def shuffled_indices(size, seed=_DATASET_SEED):
  rng = random.Random(seed)
  indices = array.array(pyu.array_code(size), range(size))
  rng.shuffle(indices)

  return indices


def shuffled_data(data, seed=None):
  indices = shuffled_indices(len(data), seed=seed)

  return type(data)(data[i] for i in indices)

