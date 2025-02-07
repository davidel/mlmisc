import py_misc_utils.assert_checks as tas
import torch
import torch.nn.functional as F

from . import dataset_base as dsb


SEQUENCE = 'sequence'
TOKEN = 'token'
MODES = {SEQUENCE, TOKEN}


class SequenceDatasetBase:

  def __init__(self, data, context_size, pad=None, mode=SEQUENCE):
    tas.check(mode in MODES, msg=f'Sequence mode ({mode}) must be one of {MODES}')

    pad_size = sum(pad['pad']) if pad is not None else 0

    self._data = data
    self._context_size = context_size - pad_size
    self._pad = pad
    self._mode = mode

  def extra_arg(self, name):
    extra_arg = getattr(self._data, 'extra_arg', None)

    return extra_arg(name) if extra_arg is not None else None

  def _sample(self, tokens, idx):
    offset = idx + self._context_size
    ybase = idx + 1 if self._mode == SEQUENCE else offset

    return tokens[idx: offset], tokens[ybase: offset + 1]

  def _padded(self, x, y):
    if self._pad is not None:
      x = F.pad(x, self._pad['pad'], value=self._pad['value'])

    return x, y


class SequenceDataset(dsb.Dataset, SequenceDatasetBase):

  def __init__(self, data, context_size,
               pipeline=None,
               pad=None,
               mode='sequence',
               **kwargs):
    dsb.Dataset.__init__(self, pipeline=pipeline, **kwargs)
    SequenceDatasetBase.__init__(self, data, context_size, pad=pad, mode=mode)

  def __len__(self):
    return max(len(self._data) - self._context_size, 0)

  def get_sample(self, i):
    x, y = self._sample(self._data, i)

    return self._padded(x, y)


class SequenceIterDataset(dsb.IterableDataset, SequenceDatasetBase):

  def __init__(self, data, context_size,
               pipeline=None,
               tokenizer=None,
               pad=None,
               mode=SEQUENCE,
               **kwargs):
    dsb.IterableDataset.__init__(self, pipeline=pipeline, tokenizer=tokenizer, **kwargs)
    SequenceDatasetBase.__init__(self, data, context_size, pad=pad, mode=mode)
    self._tokenizer = tokenizer

  def enum_samples(self):
    tokens = []
    for data in self._data:
      if isinstance(data, str):
        tokens.extend(self._tokenizer.encode(data))
      else:
        tokens.extend(data)

      for i in range(len(tokens) - self._context_size):
        x, y = self._sample(tokens, i)
        x, y = torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

        yield self._padded(x, y)

      tokens = tokens[-self._context_size:]

