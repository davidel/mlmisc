import py_misc_utils.alog as alog
import py_misc_utils.gfs as gfs
import py_misc_utils.pipeline as pypl
import torch

from . import core_utils as cu
from . import dataset_base as dsb


class TokensDataset(dsb.Dataset):

  def __init__(self, data, window_size, mode,
               pipeline=None,
               target_transform=None,
               **kwargs):
    super().__init__(pipeline=pipeline, **kwargs)
    self._data = data
    self._window_size = window_size
    self._mode = mode

  def __len__(self):
    return max(len(self._data) - 2 * self._window_size, 0)

  def get_sample(self, i):
    mid, eow = i + self._window_size, i + 2 * self._window_size + 1

    wnd = torch.cat((self._data[i: mid], self._data[mid + 1: eow]))
    tok = self._data[mid: mid + 1]

    if self._mode == 'cbow':
      x, y = wnd, tok
    elif self._mode == 'skipgram':
      x, y = tok, wnd
    else:
      alog.xraise(ValueError, f'Invalid mode: {self._mode}')

    return x, y


def create(tokens_path, window_size, mode,
           split_pct=0.9,
           **kwargs):
  tokens = cu.torch_load(gfs.normpath(tokens_path))

  train_limit = int(len(tokens) * split_pct)
  train_data = tokens[: train_limit]
  test_data = tokens[train_limit:]

  # We used torch.int in tkz.tokenize_data() above to reduce the memory footprint,
  # but some PyTorch APIs require torch.long (!?!) so we convert them on the fly.
  to_long = dsb.to_transform(dtype=torch.long)
  pipeline = pypl.Pipeline(dsb.transformer(sample=to_long, target=to_long))

  kwargs['pipeline'] = pipeline

  train_dataset = TokensDataset(train_data, window_size, mode, **kwargs)
  test_dataset = TokensDataset(test_data, window_size, mode, **kwargs)

  return dict(train=train_dataset, test=test_dataset)

