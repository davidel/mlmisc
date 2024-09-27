import torch
import torch.nn.functional as F

from . import dataset_utils as dsu


class NextTokenDataset(torch.utils.data.Dataset):

  def __init__(self, data, context_size, pad=None):
    pad_size = sum(pad['pad']) if pad is not None else 0

    super().__init__()
    self.data = data
    self.context_size = context_size - pad_size
    self.pad = pad

  def __len__(self):
    return max(len(self.data) - self.context_size, 0)

  def __getitem__(self, i):
    if isinstance(i, slice):
      return dsu.sub_dataset(self, i)

    offset = i + self.context_size
    x, y = self.data[i: offset], self.data[offset: offset + 1]

    if self.pad is not None:
      x = F.pad(x, self.pad['pad'], value=self.pad['value'])

    return x, y

