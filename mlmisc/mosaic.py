import collections
import bisect
import math

import py_misc_utils.alog as alog
import py_misc_utils.assert_checks as tas
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import layer_utils as lu


class TilesPod(nn.Module):

  def __init__(self, msize, count, dtype=None, init=None):
    weight = torch.empty(msize, count * msize, dtype=dtype)
    (init or nn.init.kaiming_uniform_)(weight)

    super().__init__()
    self.weight = nn.Parameter(weight)
    self.reset()

  def reset(self):
    self.idx = 0
    self.used = 0

  def get_indices(self, n):
    msize, wsize = self.weight.shape
    count = wsize // msize

    indices, left = [], n
    while left > 0:
      size = min(left, count - self.idx)
      indices.append((self.idx, self.idx + size))
      self.idx = (self.idx + size) % count
      left -= size

    self.used += count

    return indices

  def buil_row(self, indices):
    msize, wsize = self.weight.shape

    row_parts = []
    for start, end in indices:
      row_parts.append(self.weight[:, start: end])

    return torch.hstack(row_parts)

  def build_mosaic(self, parts):
    col_parts = [self.buil_row(indices) for indices in parts]

    return torch.vstack(col_parts)

  def stats(self):
    return dict(used=self.used, nparams=self.weight.numel())


class MosaicManager:

  def __init__(self, mods_budget, dtype=None, div_factor=None, init=None):
    self.mods_budget = mods_budget
    self.dtype = dtype
    self.init = init
    self.div_factor = div_factor or 16
    self.mods = dict()

  def reset(self):
    for mod in self.mods.values():
      mod.reset()

  def module_size(self, idim, odim, msize=None):
    if msize is None:
      msize = round(max(idim, odim) / self.div_factor)
      sizes = sorted(self.mods_budget.keys())
      x = bisect.bisect_left(sizes, msize)
      msize = sizes[min(x, len(sizes) - 1)]

    alog.debug0(f'Selected block size {msize} for layer of size {idim}x{odim}')

    return msize

  def get(self, n):
    budget = self.mods_budget.get(n)
    tas.check_is_not_none(budget,
                          msg=f'Unlisted module size {n}: ' \
                          f'available={list(self.mods_budget.keys())}')

    mod = self.mods.get(n)
    if mod is None:
      mod = TilesPod(n, budget, dtype=self.dtype, init=self.init)
      self.mods[n] = mod

    return mod

  def stats(self):
    stats = dict()
    for n, mod in self.mods.items():
      stats[n] = mod.stats()

    return stats

  def build_modules(self, msize, icount, ocount):
    mod = self.get(msize)

    parts = []
    for i in range(icount):
      parts.append(mod.get_indices(ocount))

    return mod, parts


class Mosaic(nn.Module):

  def __init__(self, idim, odim, mmgr,
               msize=None,
               post=None,
               bias=True,
               pad_value=None):
    msize = mmgr.module_size(idim, odim, msize=msize)
    icount = (idim + msize - 1) // msize
    ocount = (odim + msize - 1) // msize
    rem = idim % msize

    super().__init__()
    self.odim = odim
    self.post = lu.create(post or nn.Identity)
    if rem != 0:
      self.pad = lambda x: F.pad(x, (0, msize - rem), value=pad_value)
    else:
      self.pad = lambda x: x
    self.mod, self.parts = mmgr.build_modules(msize, icount, ocount)
    if bias:
      bound = 1.0 / math.sqrt(odim)
      weight = torch.empty(odim, dtype=mmgr.dtype).uniform_(-bound, bound)
      self.bias = nn.Parameter(weight)
    else:
      self.bias = 0
    self.fc_mat = None

  def _get_fc_mat(self):
    if self.training:
      self.fc_mat = None
      fc_mat = self.mod.build_mosaic(self.parts)
    else:
      fc_mat = self.fc_mat
      if fc_mat is None:
        with torch.no_grad():
          self.fc_mat = fc_mat = self.mod.build_mosaic(self.parts)

    return fc_mat

  def forward(self, x):
    fc_mat = self._get_fc_mat()

    x = self.pad(x)
    x = x @ fc_mat
    x = x[..., : self.odim]
    x = self.post(x + self.bias)

    return x

