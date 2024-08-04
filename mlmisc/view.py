import collections

import numpy as np
import py_misc_utils.assert_checks as tas
import torch
import torch.nn as nn


Seek = collections.namedtuple('Seek', 'n')
Keep = collections.namedtuple('Keep', 'n', defaults=(1,))
Join = collections.namedtuple('Join', 'n', defaults=(2,))
Split = collections.namedtuple('Split', 'n')


def get_view_args(shape, dims):
  args, pos = [], 0
  for d in dims:
    if isinstance(d, int):
      args.append(d)
    elif isinstance(d, Seek):
      pos = d.n
    elif isinstance(d, Keep):
      for i in range(d.n):
        args.append(shape[pos + i])
      pos += d.n
    elif isinstance(d, Join):
      args.append(np.prod(shape[d.s: d.s + d.n]))
      pos += d.n
    elif isinstance(d, Split):
      tas.check_eq(shape[pos] % abs(d.n), 0,
                   msg=f'Split dimension must be multiple: shape={shape} ' \
                   f'dim={shape[pos]} n={d.n}')
      if d.n > 0:
        args.append(d.n)
        args.append(shape[pos] // d.n)
      else:
        args.append(shape[pos] // -d.n)
        args.append(-d.n)

      pos += 1

  return args


class View(nn.Module):

  def __init__(self, *dims):
    super().__init__()
    self.dims = dims

  def forward(self, x):
    vdims = get_view_args(x.shape, self.dims)

    return x.view(*vdims)

