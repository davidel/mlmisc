import collections
import math

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
    self.used = 0


class ModMat(nn.Module):

  def __init__(self, n, dtype=None):
    super().__init__()
    weight = torch.empty(n, n, dtype=dtype)
    nn.init.kaiming_uniform_(weight)
    self.weight = nn.Parameter(weight)


class TinyModManager:

  def __init__(self, max_params, dtype=None):
    self.max_params = max_params
    self.dtype = dtype
    self.mods = collections.defaultdict(_Mods)

  def _total_params(self):
    return sum(mod.params for mod in self.mods.values())

  def reset(self):
    for mod in self.mods.values():
      mod.idx = 0
      mod.used = 0

  def get(self, n):
    if isinstance(self.max_params, dict):
      max_params = self.max_params.get(n)
      tas.check_is_not_none(max_params,
                            msg=f'Unlisted module size {n}: ' \
                            f'available={list(self.max_params.keys())}')

      mod = self.mods[n]
      create = max_params > mod.params
    else:
      mod = self.mods[n]
      create = self.max_params > self._total_params()

    if not create:
      tas.check(mod.mods, msg=f'Cannot create module of size {n}, no available ' \
                f'budget (max_params={self.max_params})')

      m = mod.mods[mod.idx]
      mod.idx = (mod.idx + 1) % len(mod.mods)
    else:
      m = ModMat(n, dtype=self.dtype)
      mod.mods.append(m)
      mod.params += ut.count_params(m)

    mod.used += 1

    return m

  def stats(self):
    stats = dict()
    for n, mod in self.mods.items():
      stats[n] = dict(count=len(mod.mods), used=mod.used, params=mod.params)

    return stats


class TinyMod(nn.Module):

  def __init__(self, idim, odim, msize, tmgr,
               post=None,
               bias=True,
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
    if bias:
      bound = 1.0 / math.sqrt(odim)
      weight = torch.empty(odim, dtype=tmgr.dtype).uniform_(-bound, bound)
      self.bias = nn.Parameter(weight)
    else:
      self.bias = 0

  def _build_fc_mat(self):
    iparts = []
    for i in range(self.icount):
      offset = i * self.ocount
      oparts = [m.weight for m in self.mods[offset: offset + self.ocount]]
      iparts.append(torch.hstack(oparts))

    return torch.vstack(iparts)

  def forward(self, x):
    dims, idim = ut.split_dims(x.shape, 1)

    rem = idim % self.msize
    if rem != 0:
      x = F.pad(x, (0, self.msize - rem), value=self.pad_value)
    mat = self._build_fc_mat()
    x = x @ mat
    x = x[..., : self.odim]
    x = self.post(x + self.bias)

    return x

