import torch
import torch.nn.functional as F

from . import dataset_base as dsb


class NextSequenceDataset(dsb.Dataset):

  def __init__(self, data, context_size,
               pad=None,
               transform=None,
               target_transform=None,
               **kwargs):
    pad_size = sum(pad['pad']) if pad is not None else 0

    super().__init__(transform=transform,
                     target_transform=target_transform,
                     **kwargs)
    self.data = data
    self.context_size = context_size - pad_size
    self.pad = pad

  def extra_arg(self, name):
    extra_arg = getattr(self.data, 'extra_arg', None)

    return extra_arg(name) if extra_arg is not None else None

  def __len__(self):
    return max(len(self.data) - self.context_size, 0)

  def get_sample(self, i):
    offset = i + self.context_size
    x, y = self.data[i: offset], self.data[i + 1: offset + 1]

    if self.pad is not None:
      x = F.pad(x, self.pad['pad'], value=self.pad['value'])

    return x, y

