import bisect
import random

import py_misc_utils.alog as alog
import torch

from . import dataset_base as dsb


class ShufflerDataset(torch.utils.data.IterableDataset, dsb.DatasetBase):

  def __init__(self, data, buffer_size=1024):
    torch.utils.data.IterableDataset.__init__(self)
    dsb.DatasetBase.__init__(self)
    self._data = data
    self._buffer_size = buffer_size
    self.add_sources(data)

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
    return self.generate()


class TransformDataset(dsb.Dataset):

  def __init__(self, data, pipeline, **kwargs):
    dsb.Dataset.__init__(self, pipeline=pipeline, **kwargs)
    self._data = data
    self.add_sources(data)

  def get_sample(self, i):
    return self._data[i]


class IterableTransformDataset(dsb.IterableDataset):

  def __init__(self, data, pipeline, **kwargs):
    dsb.IterableDataset.__init__(self, pipeline=pipeline, **kwargs)
    self._data = data
    self.add_sources(data)

  def enum_samples(self):
    yield from self._data


class JoinedDatasetBase:

  def __init__(self, datasets):
    self._datasets = tuple(datasets)

  def bases(self):
    return self._datasets

  def extra_arg(self, name):
    args = []
    for data in self._datasets:
      extra_arg_fn = getattr(data, 'extra_arg', None)
      if callable(extra_arg_fn):
        args.append(extra_arg_fn(name))
      else:
        args.append(None)

    if all(args[0] == arg for arg in args):
      return args[0] if args else None

    return args


class JoinedDataset(torch.utils.data.Dataset, JoinedDatasetBase):

  def __init__(self, datasets):
    torch.utils.data.Dataset.__init__(self)
    JoinedDatasetBase.__init__(self, datasets)
    self._sizes = [0];
    for data in datasets:
      self._sizes.append(self._sizes[-1] + len(data))

  def __getitem__(self, i):
    pos = bisect.bisect_right(self._sizes, i) - 1
    data = self._datasets[pos]

    return data[i - self._sizes[pos]]

  def __len__(self):
    return self._sizes[-1]


class IterableJoinedDataset(torch.utils.data.IterableDataset, JoinedDatasetBase):

  def __init__(self, datasets):
    torch.utils.data.IterableDataset.__init__(self)
    JoinedDatasetBase.__init__(self, datasets)

  def __len__(self):
    size = 0
    for data in self._datasets:
      len_fn, csize = getattr(data, '__len__', None), None
      if callable(len_fn):
        csize = len_fn()

      if csize is None:
        return None

      size += csize

    return size

  def generate(self):
    for data in self._datasets:
      yield from data

  def __iter__(self):
    return self.generate()

