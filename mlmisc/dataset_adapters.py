import random

import py_misc_utils.alog as alog
import torch

from . import dataset_base as dsb


class ShufflerDataset(torch.utils.data.IterableDataset):

  def __init__(self, data, buffer_size=1024):
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


class TransformDataset(dsb.Dataset):

  def __init__(self, data, pipeline, **kwargs):
    super().__init__(pipeline=pipeline, **kwargs)
    self._data = data

  def get_sample(self, i):
    return self._data[i]

  def __len__(self):
    return len(self._data)


class IterableTransformDataset(dsb.IterableDataset):

  def __init__(self, data, pipeline, **kwargs):
    super().__init__(pipeline=pipeline, **kwargs)
    self._data = data

  def enum_samples(self):
    yield from self._data

