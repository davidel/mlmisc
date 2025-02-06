import torch
import torch.nn.functional as F

from . import dataset_base as dsb


class SequenceDataset(dsb.Dataset):

  def __init__(self, data, context_size,
               pipeline=None,
               pad=None,
               mode='sequence',
               **kwargs):
    pad_size = sum(pad['pad']) if pad is not None else 0

    super().__init__(pipeline=pipeline, **kwargs)
    self._data = data
    self._context_size = context_size - pad_size
    self._pad = pad
    self._mode = mode

  def extra_arg(self, name):
    extra_arg = getattr(self._data, 'extra_arg', None)

    return extra_arg(name) if extra_arg is not None else None

  def __len__(self):
    return max(len(self._data) - self._context_size, 0)

  def get_sample(self, i):
    offset = i + self._context_size
    ybase = i + i if self._mode == 'sequence' else offset
    x, y = self._data[i: offset], self._data[ybase: offset + 1]

    if self._pad is not None:
      x = F.pad(x, self._pad['pad'], value=self._pad['value'])

    return x, y


class SequenceIterDataset(dsb.IterableDataset):

  def __init__(self, data, context_size,
               pipeline=None,
               tokenizer=None,
               pad=None,
               mode='sequence',
               **kwargs):
    pad_size = sum(pad['pad']) if pad is not None else 0

    super().__init__(pipeline=pipeline, tokenizer=tokenizer, **kwargs)
    self._data = data
    self._tokenizer = tokenizer
    self._context_size = context_size - pad_size
    self._pad = pad
    self._mode = mode

  def extra_arg(self, name):
    extra_arg = getattr(self._data, 'extra_arg', None)

    return extra_arg(name) if extra_arg is not None else None

  def enum_samples(self):
    tokens = []
    for data in self._data:
      if isinstance(data, str):
        tokens.extend(self._tokenizer.encode(data))
      else:
        tokens.extend(data)

      for i in range(len(tokens) - self._context_size):
        offset = i + self._context_size
        ybase = i + i if self._mode == 'sequence' else offset
        x, y = tokens[i: offset], tokens[ybase: offset + 1]

        x, y = torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

        if self._pad is not None:
          x = F.pad(x, self._pad['pad'], value=self._pad['value'])

        yield x, y

      tokens = tokens[-self._context_size:]

