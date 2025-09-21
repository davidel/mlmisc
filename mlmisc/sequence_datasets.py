import bisect
import copy

import py_misc_utils.alog as alog
import py_misc_utils.assert_checks as tas
import py_misc_utils.num_utils as pynu
import py_misc_utils.pipeline as pypl
import torch
import torch.nn.functional as F

from . import dataset_base as dsb


class TokenSampler:

  def __init__(self, window_size, **kwargs):
    self.context_size = window_size + 1
    self._window_size = window_size
    self.allows_padding = False

  def __call__(self, data, idx):
    offset = idx + self._window_size

    return data[idx: offset], data[offset: offset + 1]


class SequenceSampler:

  def __init__(self, window_size, **kwargs):
    self.context_size = window_size + 1
    self.allows_padding = True

  def __call__(self, data, idx):
    window_size = min(len(data) - idx, self.context_size) - 1
    offset = idx + window_size

    return data[idx: offset], data[idx + 1: offset + 1]


class CbowSampler:

  def __init__(self, window_size, **kwargs):
    self.context_size = 2 * window_size + 1
    self._window_size = window_size
    self.allows_padding = False

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

def _get_sampler(mode, window_size, **kwargs):
  tas.check_in(mode, set(_SAMPLERS.keys()),
               msg=f'Invalid mode')

  return _SAMPLERS[mode](window_size, **kwargs)


class SequenceDataset(dsb.Dataset):

  def __init__(self, data, context_size, mode, pipeline=None, **kwargs):
    dsb.Dataset.__init__(self, pipeline=pipeline, **kwargs)
    self._data = data
    self._sampler = _get_sampler(mode, context_size, **kwargs)
    self.add_sources(data)

  def __len__(self):
    return max(len(self._data) + 1 - self._sampler.context_size, 0)

  def get_sample(self, i):
    return self._sampler(self._data, i)


class SequenceProcessor(pypl.IterElement):

  def __init__(self, context_size, mode, tokenizer,
               batch_size=None,
               min_context_size=4,
               num_context_buckets=None,
               **kwargs):
    super().__init__()
    self._sampler = _get_sampler(mode, context_size, **kwargs)
    self._tokenizer = tokenizer
    self._batch_size = batch_size
    self._min_context_size = min_context_size
    self._num_context_buckets = num_context_buckets
    self._pad_id = tokenizer.pad_id()
    if self._pad_id is None:
      self._pad_id = tokenizer.eos_id()
    self._reset()

  def _reset(self):
    self._context_buckets = dict()
    self._bucket_sizes = []
    if self._num_context_buckets is not None:
      step = ((self._sampler.context_size - self._min_context_size) //
               self._num_context_buckets)

      self._bucket_sizes = list(range(self._min_context_size, self._sampler.context_size, step))

      margin = self._sampler.context_size - self._bucket_sizes[-1]
      if margin > step // 2:
        self._bucket_sizes.append(self._sampler.context_size)
      else:
        self._bucket_sizes[-1] = self._sampler.context_size

  def _tokenize(self, data):
    if isinstance(data, str):
      tdata = self._tokenizer.encode(data)
    elif isinstance(data, bytes):
      tdata = self._tokenizer.encode(data.decode())
    else:
      tdata = data

    return tdata

  def _get_bucket_size(self, size):
    pos = bisect.bisect_right(self._bucket_sizes, size)

    return self._bucket_sizes[pos] if len(self._bucket_sizes) > pos else None

  def _enqueue(self, data):
    if len(data) >= self._sampler.context_size:
      size = self._sampler.context_size
    elif self._sampler.allows_padding:
      if self._num_context_buckets is not None:
        size = self._get_bucket_size(len(data))
      else:
        size = self._sampler.context_size

      data = list(data) + [self._pad_id] * (size - len(data))
    else:
      alog.debug(f'Discarded sequence of length {len(data)}')
      return

    if (bucket := self._context_buckets.get(size)) is None:
      self._context_buckets[size] = bucket = []

    max_index = max(1, len(data) + 1 - size)
    for i in range(max_index):
      bucket.append(self._sampler(data, i))

    alog.debug(f'Sequence context bucket slot {size} has {len(bucket)} samples')

    return bucket, size

  def __call__(self, data):
    for idata in data:
      tdata = self._tokenize(idata)
      result = self._enqueue(tdata)

      if result is not None:
        bucket, size = result
        if self._batch_size is None:
          for bdata in bucket:
            yield bdata

          self._context_buckets[size] = []
        else:
          batches = []
          for i in range(0, len(bucket), self._batch_size):
            if len(bucket) >= i + self._batch_size:
              batches.append(bucket[i: i + self._batch_size])
            else:
              break

          self._context_buckets[size] = bucket[i:]

          for batch in batches:
            x, y = [b[0] for b in batch], [b[1] for b in batch]
            yield x, y

  def flush(self, data):
    yield from self(data)

    for size, bucket in self._context_buckets.items():
      if self._batch_size is None:
        for bdata in bucket:
          yield bdata
      elif bucket:
        x, y = [b[0] for b in bucket], [b[1] for b in bucket]
        yield x, y

    self._reset()

  def clone(self):
    new_self = copy.copy(self)
    new_self._reset()

    return new_self


class Padder(pypl.IterElement):

  def __init__(self, pad):
    super().__init__()
    self._pad = pad

  def __call__(self, data):
    for idata in data:
      x, y = idata

      yield F.pad(x, self._pad['pad'], value=self._pad['value']), y

