import array
import functools
import random

import py_misc_utils.alog as alog
import py_misc_utils.core_utils as pycu
import py_misc_utils.pipeline as pypl
import py_misc_utils.utils as pyu
import torch


class DatasetBase:

  def __init__(self, data=None, pipeline=None, **kwargs):
    self._data = data
    self._pipeline = pyu.value_or(pipeline, pypl.Pipeline())
    self._kwargs = kwargs

  def _sources(self):
    sources = [super()]
    if self._data is not None:
      sources.append(self._data)

    return tuple(sources)

  def extra_arg(self, name):
    value = self._kwargs.get(name)
    if value is not None:
      return value

    for source in self._sources():
      extra_arg_fn = getattr(source, 'extra_arg', None)
      if callable(extra_arg_fn) and (value := extra_arg_fn(name)) is not None:
        return value

    return getattr(self._data, name, None)

  def add_extra_arg(self, name, value):
    self._kwargs[name] = value

  def __len__(self):
    for source in self._sources():
      data_length = getattr(source, '__len__', None)
      if data_length is not None:
        return data_length()

    return self.extra_arg('size')

  def process_sample(self, data):
    return self._pipeline(data)

  def pipeline(self):
    return self._pipeline.clone()


class Dataset(torch.utils.data.Dataset, DatasetBase):

  def __init__(self, data=None, pipeline=None, **kwargs):
    torch.utils.data.Dataset.__init__(self)
    DatasetBase.__init__(self, data=data, pipeline=pipeline, **kwargs)

  def __getitem__(self, i):
    if isinstance(i, slice):
      return sliced_dataset(self, i)

    data = self.get_sample(i)

    return self.process_sample(data)


class IterableDataset(torch.utils.data.IterableDataset, DatasetBase):

  def __init__(self, data=None, pipeline=None, **kwargs):
    torch.utils.data.IterableDataset.__init__(self)
    DatasetBase.__init__(self, data=data, pipeline=pipeline, **kwargs)

  def generate(self):
    for data in self.enum_samples():
      pdata = self.process_sample(data)
      if pycu.is_iterator(pdata):
        yield from pdata
      else:
        yield pdata

  def __iter__(self):
    return self.generate()


class SubDataset(torch.utils.data.Dataset, DatasetBase):

  def __init__(self, data, indices, **kwargs):
    torch.utils.data.Dataset.__init__(self)
    DatasetBase.__init__(self, data=data, **kwargs)
    self._indices = indices

  def __len__(self):
    return len(self._indices)

  def __getitem__(self, i):
    if isinstance(i, slice):
      return SubDataset(self._data, self._indices[i])

    return self._data[self._indices[i]]


def to_transform_fn(x, **kwargs):
  return torch.as_tensor(x, **kwargs)


def to_transform(**kwargs):
  return functools.partial(to_transform_fn, **kwargs)


def transformer_fn(sample, target, x):
  s, t = x

  return sample(s), target(t)


def transformer(sample=pycu.ident, target=pycu.ident):
  return functools.partial(transformer_fn, sample, target)


def items_selector_fn(items, x):
  return [x[i] for i in items] if isinstance(items, (list, tuple)) else x[items]


def items_selector(items):
  return functools.partial(items_selector_fn, items)


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

