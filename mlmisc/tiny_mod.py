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


class TinyModManager:

  def __init__(self, max_params, bias=True):
    self.max_params = max_params
    self.bias = bias
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
      m = nn.Linear(n, n, bias=self.bias)
      mod.mods.append(m)
      mod.params += ut.count_params(m)

    self.used += 1

    return m

  def stats(self):
    return {n: mod.params for n, mod in self.mods.items()}



class TinyMod(nn.Module):

  def __init__(self, idim, odim, msize, tmgr,
               mdim=None,
               post=None,
               pad_value=0):
    mdim = mdim or max(idim, odim)
    icount = (idim + msize - 1) // msize
    mcount = (mdim + msize - 1) // msize
    ocount = (odim + msize - 1) // msize

    super().__init__()
    self.odim = odim
    self.msize = msize
    self.pad_value = pad_value
    self.fcin = nn.Linear(icount, mcount)
    self.mods = nn.ModuleList([tmgr.get(msize) for _ in range(mcount)])
    self.fcout = nn.Linear(mcount, ocount)
    self.post = post or nn.Identity()

  def forward(self, x):
    dims, idim = ut.split_dims(x.shape, 1)

    rem = idim % self.msize
    if rem != 0:
      x = F.pad(x, (0, self.msize - rem), value=self.pad_value)
      idim += self.msize - rem

    icount = idim // self.msize
    x = torch.reshape(x, (*dims, icount, self.msize)) # (*DIMS, ICOUNT, MSIZE)
    x = ut.tail_permute(x) # (*DIMS, MSIZE, ICOUNT)

    x = self.fcin(x) # (*DIMS, MSIZE, MCOUNT)

    x = ut.tail_permute(x) # (*DIMS, MCOUNT, MSIZE)

    parts, sdim = [], x.ndim - 2
    for i, mod in enumerate(self.mods):
      yv = mod(torch.squeeze(torch.select(x, sdim, i), sdim))
      parts.append(torch.unsqueeze(yv, sdim))

    x = torch.cat(parts, dim=sdim) # (*DIMS, MCOUNT, MSIZE)
    x = ut.tail_permute(x) # (*DIMS, MSIZE, MCOUNT)

    x = self.fcout(x) # (*DIMS, MSIZE, OCOUNT)

    x = torch.reshape(x, (*dims, -1)) # (*DIMS, MSIZE * OCOUNT)
    x = x[:, : self.odim] # (*DIMS, ODIM)

    x = self.post(x)

    return x

