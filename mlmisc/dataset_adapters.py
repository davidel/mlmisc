import random

import py_misc_utils.alog as alog
import torch

from . import dataset_base as dsb


class ShufflerDataset(torch.utils.data.IterableDataset, dsb.DatasetWrapper):

  def __init__(self, data, buffer_size=1024):
    torch.utils.data.IterableDataset.__init__(self)
    dsb.DatasetWrapper.__init__(self, data)
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
    return self.generate()


class TransformDataset(dsb.DatasetWrapper, dsb.Dataset):

  def __init__(self, data, pipeline, **kwargs):
    dsb.DatasetWrapper.__init__(self, data, **kwargs)
    dsb.Dataset.__init__(self, pipeline=pipeline)

  def get_sample(self, i):
    return self._data[i]


class IterableTransformDataset(dsb.DatasetWrapper, dsb.IterableDataset):

  def __init__(self, data, pipeline, **kwargs):
    dsb.DatasetWrapper.__init__(self, data, **kwargs)
    dsb.IterableDataset.__init__(self, pipeline=pipeline)

  def enum_samples(self):
    yield from self._data

