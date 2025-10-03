import bisect
import copy
import random
import time

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
    super().__init__(pipeline=pipeline, **kwargs)
    self._data = data
    self._sampler = _get_sampler(mode, context_size, **kwargs)
    self.add_sources(data)

  def __len__(self):
    return max(len(self._data) + 1 - self._sampler.context_size, 0)

  def get_sample(self, i):
    return self._sampler(self._data, i)


class TokenizerProcessor:

  def __init__(self, tokenizer, **kwargs):
    self._tokenizer = tokenizer

  def __call__(self, data):
    if isinstance(data, str):
      tdata = self._tokenizer.encode(data)
    elif isinstance(data, bytes):
      tdata = self._tokenizer.encode(data.decode())
    else:
      tdata = data

    return tdata


class _Bucket:

  def __init__(self, size,
               shuffle_size=None):
    self.size = size
    self._shuffle_size = shuffle_size
    self._samples = []
    self.mtime = time.time()

  def __len__(self):
    return len(self._samples)

  def __iter__(self):
    if self._shuffle_size is not None:
      return iter(random.sample(self._samples, len(self._samples)))
    else:
      return iter(self._samples)

  def add_sample(self, sample):
    self._samples.append(sample)
    self.mtime = time.time()

  def clear(self):
    self._samples = []
    self.mtime = time.time()

  def get_samples(self):
    samples, self._samples = self._samples, []
    self.mtime = time.time()

    if self._shuffle_size is not None:
      random.shuffle(samples)

    return samples

  def get_batches(self, batch_size, force=False):
    batches = []
    if (self._samples and
        (force or self._shuffle_size is None or
         len(self._samples) >= self._shuffle_size)):
      if self._shuffle_size is not None:
        random.shuffle(self._samples)

      for pos in range(0, len(self._samples), batch_size):
        if len(self._samples) >= pos + batch_size:
          batches.append(self._samples[pos: pos + batch_size])
        else:
          break

      self._samples = self._samples[pos:]
      if force and self._samples:
        batches.append(self._samples)
        self._samples = []

      self.mtime = time.time()

    return batches


class SequenceProcessor(pypl.IterElement):

  def __init__(self, context_size, mode,
               batch_size=None,
               pad_id=None,
               min_context_size=0,
               num_context_buckets=None,
               flush_interval=None,
               shuffle_size=None,
               **kwargs):
    super().__init__()
    self._sampler = _get_sampler(mode, context_size, **kwargs)
    self._batch_size = batch_size
    self._min_context_size = min_context_size
    self._num_context_buckets = num_context_buckets
    self._flush_interval = flush_interval
    self._shuffle_size = shuffle_size
    self._pad_id = pad_id
    self._bucket_sizes = self._create_buckets(self._num_context_buckets,
                                              self._sampler.context_size,
                                              self._min_context_size)

    self._reset()

  @staticmethod
  def _create_buckets(count, context_size, min_context_size):
    bucket_sizes = []
    if count is not None:
      step = ((context_size - min_context_size) // count)

      bucket_sizes = list(range(min_context_size, context_size, step))

      margin = context_size - bucket_sizes[-1]
      if margin > step // 4:
        bucket_sizes.append(context_size)
      else:
        bucket_sizes[-1] = context_size

    return bucket_sizes

  def _reset(self):
    self._context_buckets = dict()

  def _get_bucket_size(self, size):
    pos = bisect.bisect_left(self._bucket_sizes, size)

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
      alog.debug3(f'Discarded sequence of length {len(data)}')
      return

    if (bucket := self._context_buckets.get(size)) is None:
      bucket = _Bucket(size, shuffle_size=self._shuffle_size)
      self._context_buckets[size] = bucket

    max_index = max(1, len(data) + 1 - size)
    for i in range(max_index):
      bucket.add_sample(self._sampler(data, i))

    alog.verbose(f'Sequence context bucket slot {size} has {len(bucket)} samples')

    return bucket

  def _collate(self, batch):
    cdata = []
    for bdata in batch:
      if len(bdata) > len(cdata):
        cdata = cdata + [[] for _ in range(len(bdata) - len(cdata))]

      for j, data in enumerate(bdata):
        cdata[j].append(data)

    return cdata

  def __call__(self, data):
    for idata in data:
      bucket = self._enqueue(idata)

      if bucket is not None:
        if self._batch_size is None:
          for bdata in bucket.get_samples():
            yield bdata
        else:
          yield from self._flush_bucket(bucket)

    yield from self._flush_buckets()

  def _flush_bucket(self, bucket, force=False):
    batches = bucket.get_batches(self._batch_size, force=force)
    for batch in batches:
      yield self._collate(batch)

  def _flush_buckets(self):
    if self._batch_size is not None and self._flush_interval is not None:
      now = time.time()

      for size, bucket in self._context_buckets.items():
        if now - bucket.mtime >= self._flush_interval:
          yield from self._flush_bucket(bucket, force=True)

  def flush(self, data):
    yield from self(data)

    for bucket in self._context_buckets.values():
      if self._batch_size is None:
        for bdata in bucket.get_samples():
          yield bdata
      else:
        yield from self._flush_bucket(bucket, force=True)

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

