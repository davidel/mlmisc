import collections
import bisect
import math

import py_misc_utils.alog as alog
import py_misc_utils.assert_checks as tas
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModsPod:

  def __init__(self):
    self.mods = []
    self.reset()

  def reset(self):
    self.idx = 0
    self.used = 0


class ModMat(nn.Module):

  def __init__(self, n, dtype=None):
    super().__init__()
    weight = torch.empty(n, n, dtype=dtype)
    nn.init.kaiming_uniform_(weight)
    self.weight = nn.Parameter(weight)


class TinyModManager:

  def __init__(self, mods_budget, dtype=None, div_factor=None):
    self.mods_budget = mods_budget
    self.dtype = dtype
    self.div_factor = div_factor or 16
    self.mods = collections.defaultdict(ModsPod)

  def _total_params(self):
    return sum(mod.params for mod in self.mods.values())

  def reset(self):
    for mod in self.mods.values():
      mod.reset()

  def module_size(self, idim, odim, msize=None):
    if msize is None:
      msize = round(max(idim, odim) / self.div_factor)
      sizes = sorted(self.mods_budget.keys())
      x = bisect.bisect_left(sizes, msize)
      msize = sizes[x]

    alog.debug0(f'Selected block size {msize} for layer of size {idim}x{odim}')

    return msize

  def get(self, n):
    budget = self.mods_budget.get(n)
    tas.check_is_not_none(budget,
                          msg=f'Unlisted module size {n}: ' \
                          f'available={list(self.mods_budget.keys())}')

    mod = self.mods[n]

    if len(mod.mods) >= budget:
      tas.check(mod.mods, msg=f'Cannot create module of size {n}, no available ' \
                f'budget (mods_budget={self.mods_budget})')

      m = mod.mods[mod.idx]
      mod.idx = (mod.idx + 1) % len(mod.mods)
    else:
      m = ModMat(n, dtype=self.dtype)
      mod.mods.append(m)

    mod.used += 1

    return m

  def stats(self):
    stats = dict()
    for n, mod in self.mods.items():
      stats[n] = dict(count=len(mod.mods),
                      used=mod.used,
                      params=len(mod.mods) * n**2)

    return stats

  def build_modules(self, msize, icount, ocount):
    mods = [self.get(msize) for _ in range(icount * ocount)]

    parts = []
    for i in range(icount):
      offset = i * ocount
      parts.append([m.weight for m in mods[offset: offset + ocount]])

    return nn.ModuleList(mods), parts


class TinyMod(nn.Module):

  def __init__(self, idim, odim, tmgr,
               msize=None,
               post=None,
               bias=True,
               pad_value=None):
    msize = tmgr.module_size(idim, odim, msize=msize)
    icount = (idim + msize - 1) // msize
    ocount = (odim + msize - 1) // msize
    rem = idim % msize

    super().__init__()
    self.odim = odim
    self.post = post or nn.Identity()
    if rem != 0:
      self.pad = lambda x: F.pad(x, (0, msize - rem), value=pad_value)
    else:
      self.pad = lambda x: x
    self.mods, self.parts = tmgr.build_modules(msize, icount, ocount)
    if bias:
      bound = 1.0 / math.sqrt(odim)
      weight = torch.empty(odim, dtype=tmgr.dtype).uniform_(-bound, bound)
      self.bias = nn.Parameter(weight)
    else:
      self.bias = 0

  def forward(self, x):
    x = self.pad(x)

    mat = torch.vstack([torch.hstack(oparts) for oparts in self.parts])

    x = x @ mat
    x = x[..., : self.odim]
    x = self.post(x + self.bias)

    return x

