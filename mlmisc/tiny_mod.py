import collections

import py_misc_utils.alog as alog
import py_misc_utils.assert_checks as tas
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import utils as ut


class _Mods:

  def __init__(self):
    self.idx = 0
    self.params = 0
    self.mods = []


class ModMat(nn.Module):

  def __init__(self, n, dtype=None):
    super().__init__()
    self.weight = nn.Parameter(torch.randn(n, n, dtype=dtype))


class TinyModManager:

  def __init__(self, max_params):
    self.max_params = max_params
    self.mods = collections.defaultdict(_Mods)
    self.used = 0

  def _total_params(self):
    return sum(mod.params for mod in self.mods.values())

  def reset(self):
    self.used = 0
    for mod in self.mods.values():
      mod.idx = 0

  def get(self, n):
    if isinstance(self.max_params, dict):
      max_params = self.max_params.get(n, None)
      tas.check_is_not_none(max_params,
                            msg=f'Unlisted module size {n}: ' \
                            f'available={list(self.max_params.keys())}')

      mod = self.mods[n]
      create = max_params > mod.params
    else:
      mod = self.mods[n]
      create = self.max_params > self._total_params()

    if not create:
      tas.check(mod.mods, msg=f'Cannot create module of size {n}, no available budget')

      m = mod.mods[mod.idx]
      mod.idx = (mod.idx + 1) % len(mod.mods)
    else:
      m = ModMat(n)
      mod.mods.append(m)
      mod.params += ut.count_params(m)

    self.used += 1

    return m

  def stats(self):
    return {n: mod.params for n, mod in self.mods.items()}



class TinyMod(nn.Module):

  def __init__(self, idim, odim, msize, tmgr,
               post=None,
               pad_value=None):
    super().__init__()
    self.idim = idim
    self.odim = odim
    self.msize = msize
    self.post = post or nn.Identity()
    self.pad_value = pad_value or 0
    self.icount = (idim + msize - 1) // msize
    self.ocount = (odim + msize - 1) // msize
    self.mods = nn.ModuleList([tmgr.get(msize) for _ in range(self.icount * self.ocount)])

  def _build_fc_mat(self):
    isize, osize = self.icount * self.msize, self.ocount * self.msize
    mat = torch.empty(isize, osize)
    for i in range(self.icount):
      for o in range(self.ocount):
        idx = i * self.ocount + o
        mod = self.mods[idx]
        ioffset, ooffset = i * self.msize, o * self.msize
        mat[ioffset: ioffset + self.msize, ooffset: ooffset + self.msize] = mod.weight

    return mat

  def forward(self, x):
    dims, idim = ut.split_dims(x.shape, 1)

    rem = idim % self.msize
    if rem != 0:
      x = F.pad(x, (0, self.msize - rem), value=self.pad_value)

    mat = self._build_fc_mat()
    x = x @ mat

    x = x[..., : self.odim] # (*DIMS, ODIM)

    x = self.post(x)

    return x

