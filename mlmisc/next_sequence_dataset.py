import torch
import torch.nn.functional as F

from . import dataset_utils as dsu


class NextSequenceDataset(torch.utils.data.Dataset):

  def __init__(self, data, context_size,
               pad=None,
               ydtype=None,
               **kwargs):
    pad_size = sum(pad['pad']) if pad is not None else 0

    super().__init__()
    self.data = data
    self.context_size = context_size - pad_size
    self.pad = pad
    self.ydtype = ydtype
    self.kwargs = kwargs

  def extra_arg(self, name):
    xarg = getattr(self.data, 'extra_arg', None)

    return self.kwargs.get(name) if xarg is None else xarg(name)

  def __len__(self):
    return max(len(self.data) - self.context_size, 0)

  def __getitem__(self, i):
    if isinstance(i, slice):
      return dsu.sub_dataset(self, i)

    offset = i + self.context_size
    x, y = self.data[i: offset], self.data[i + 1: offset + 1]

    if self.pad is not None:
      x = F.pad(x, self.pad['pad'], value=self.pad['value'])
    if self.ydtype is not None:
      y = y.to(dtype=self.ydtype)

    return x, y

