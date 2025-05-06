import py_misc_utils.assert_checks as tas
import torch
import torch.nn.functional as F

from . import dataset_base as dsb


class TokenSampler:

  def __init__(self, window_size):
    self.context_size = window_size + 1
    self._window_size = window_size

  def __call__(self, data, idx):
    offset = idx + self._window_size

    return data[idx: offset], data[offset: offset + 1]


class SequenceSampler:

  def __init__(self, window_size):
    self.context_size = window_size + 1
    self._window_size = window_size

  def __call__(self, data, idx):
    offset = idx + self._window_size

    return data[idx: offset], data[idx + 1: offset + 1]


class CbowSampler:

  def __init__(self, window_size):
    self.context_size = 2 * window_size + 1
    self._window_size = window_size

  def __call__(self, data, idx):
    mid, eow = idx + self._window_size, idx + self.context_size

    wnd = data[idx: mid] + data[mid + 1: eow]
    tok = data[mid: mid + 1]

    return wnd, tok


class SkipgramSampler(CbowSampler):

  def __call__(self, data, idx):
    wnd, tok = super()(data, idx)

    return tok, wnd


TOKEN = 'token'
SEQUENCE = 'sequence'
CBOW = 'cbow'
SKIPGRAM = 'skipgram'

_SAMPLERS = {
  TOKEN: TokenSampler,
  SEQUENCE: SequenceSampler,
  CBOW: CbowSampler,
  SKIPGRAM: SkipgramSampler,
}


class SequenceDatasetBase:

  def __init__(self, data, context_size, mode):
    tas.check_in(mode, set(_SAMPLERS.keys()),
                 msg=f'Invalid mode')

    self._sampler = _SAMPLERS[mode](context_size)
    self._data = data
    self._context_size = self._sampler.context_size
    self._mode = mode

  def _sample(self, data, idx):
    return self._sampler(data, idx)


class SequenceDataset(dsb.Dataset, SequenceDatasetBase):

  def __init__(self, data, context_size, mode,
               pipeline=None,
               **kwargs):
    dsb.Dataset.__init__(self, pipeline=pipeline, **kwargs)
    SequenceDatasetBase.__init__(self, data, context_size, mode)

  def __len__(self):
    return max(len(self._data) + 1 - self._context_size, 0)

  def get_sample(self, i):
    return self._sample(self._data, i)


class IterableSequenceDataset(dsb.IterableDataset, SequenceDatasetBase):

  def __init__(self, data, context_size, mode,
               pipeline=None,
               tokenizer=None,
               **kwargs):
    dsb.IterableDataset.__init__(self, pipeline=pipeline, tokenizer=tokenizer, **kwargs)
    SequenceDatasetBase.__init__(self, data, context_size, mode)
    self._tokenizer = tokenizer

  def enum_samples(self):
    tokens = []
    for data in self._data:
      if isinstance(data, str):
        tokens.extend(self._tokenizer.encode(data))
      elif isinstance(data, bytes):
        tokens.extend(self._tokenizer.encode(data.decode()))
      else:
        tokens.extend(data)

      for i in range(len(tokens) + 1 - self._context_size):
        x, y = self._sample(tokens, i)

        yield x, y

      tokens = tokens[-self._context_size + 1:]


class Padder:

  def __init__(self, pad):
    self._pad = pad

  def __call__(self, data):
    x, y = data

    return F.pad(x, self._pad['pad'], value=self._pad['value']), y

