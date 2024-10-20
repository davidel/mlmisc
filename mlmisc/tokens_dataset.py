import py_misc_utils.alog as alog
import py_misc_utils.utils as pyu
import torch

from . import dataset_base as dsb
from . import utils as ut


class TokensDataset(dsb.Dataset):

  def __init__(self, data, window_size, mode,
               transform=None,
               target_transform=None,
               **kwargs):
    super().__init__(transform=transform,
                     target_transform=target_transform,
                     **kwargs)
    self.data = data
    self.window_size = window_size
    self.mode = mode

  def __len__(self):
    return max(len(self.data) - 2 * self.window_size, 0)

  def get_sample(self, i):
    mid, eow = i + self.window_size, i + 2 * self.window_size + 1

    wnd = torch.cat((self.data[i: mid], self.data[mid + 1: eow]))
    tok = self.data[mid: mid + 1]

    if self.mode == 'cbow':
      x, y = wnd, tok
    elif self.mode == 'skipgram':
      x, y = tok, wnd
    else:
      alog.xraise(ValueError, f'Invalid mode: {self.mode}')

    return x, y


def create(tokens_path, window_size, mode,
           split_pct=None,
           **kwargs):
  split_pct = pyu.value_or(split_pct, 0.9)

  tokens = ut.torch_load(pyu.normpath(tokens_path))

  train_limit = int(len(tokens) * split_pct)
  train_data = tokens[: train_limit]
  test_data = tokens[train_limit:]

  kwargs.update(transform=dsb.to_transform(dtype=torch.long),
                target_transform=dsb.to_transform(dtype=torch.long))

  train_dataset = TokensDataset(train_data, window_size, mode, **kwargs)
  test_dataset = TokensDataset(test_data, window_size, mode, **kwargs)

  return dict(train=train_dataset, test=test_dataset)

