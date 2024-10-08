import array

import py_misc_utils.alog as alog
import py_misc_utils.assert_checks as tas
import py_misc_utils.utils as pyu
import torch

from . import utils as ut


class Dataset(torch.utils.data.Dataset):

  def __init__(self, select_fn=None, transform=None, target_transform=None, **kwargs):
    super().__init__()
    self.select_fn = select_fn or ident_select
    self.transform = transform or no_transform
    self.target_transform = target_transform or no_transform
    self.kwargs = kwargs

  def extra_arg(self, name):
    xarg = self.kwargs.get(name)
    if xarg is None:
      extra_arg = getattr(super(), 'extra_arg', None)
      if extra_arg is not None:
        xarg = extra_arg(name)

    return xarg

  def __getitem__(self, i):
    if isinstance(i, slice):
      return sliced_dataset(self, i)

    data = self.get_sample(i)
    x, y = self.select_fn(data)

    return self.transform(x), self.target_transform(y)


def no_transform(x):
  return x


def to_transform(**kwargs):
  def transform(x):
    return x.to(**kwargs)

  return transform


def ident_select(x):
  return x


def guess_select(x):
  if isinstance(x, (list, tuple)):
    return x[: 2]
  if isinstance(x, dict):
    return tuple(x.values())[: 2]

  return x


def sliced_dataset(ds, dslice):
  ds_size = len(ds)
  indices = array.array(pyu.array_code(ds_size), range(*dslice.indices(ds_size)))

  return torch.utils.data.Subset(ds, indices)

