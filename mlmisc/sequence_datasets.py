import copy

import py_misc_utils.assert_checks as tas
import py_misc_utils.pipeline as pypl
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

def _get_sampler(mode, context_size):
  tas.check_in(mode, set(_SAMPLERS.keys()),
               msg=f'Invalid mode')

  return _SAMPLERS[mode](context_size)


class SequenceDataset(dsb.Dataset):

  def __init__(self, data, context_size, mode, pipeline=None, **kwargs):
    dsb.Dataset.__init__(self, pipeline=pipeline, **kwargs)
    self._data = data
    self._sampler = _get_sampler(mode, context_size)
    self._context_size = self._sampler.context_size
    self.add_sources(data)

  def __len__(self):
    return max(len(self._data) + 1 - self._context_size, 0)

  def get_sample(self, i):
    return self._sampler(self._data, i)


class SequenceProcessor(pypl.IterElement):

  def __init__(self, context_size, mode, tokenizer):
    super().__init__()
    self._sampler = _get_sampler(mode, context_size)
    self._context_size = self._sampler.context_size
    self._tokenizer = tokenizer
    self._tokens = []

  def _process(self, data):
    for idata in data:
      if isinstance(idata, str):
        self._tokens.extend(self._tokenizer.encode(idata))
      elif isinstance(idata, bytes):
        self._tokens.extend(self._tokenizer.encode(idata.decode()))
      else:
        self._tokens.extend(idata)

      for i in range(len(self._tokens) + 1 - self._context_size):
        x, y = self._sampler(self._tokens, i)

        yield x, y

      self._tokens = self._tokens[len(self._tokens) - self._context_size + 1:]

  def clone(self):
    new_self = copy.copy(self)
    new_self._tokens = []

    return new_self


class Padder(pypl.IterElement):

  def __init__(self, pad):
    super().__init__()
    self._pad = pad

  def _process(self, data):
    for idata in data:
      x, y = idata

      yield F.pad(x, self._pad['pad'], value=self._pad['value']), y

