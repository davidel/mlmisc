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

  def reset(self):
    for mod in self.mods.values():
      mod.idx = 0

  def get(self, n):
    if isinstance(self.max_params, dict):
      max_params = self.max_params.get(n, 0)
    else:
      max_params = self.max_params

    mod = self.mods[n]

    if mod.params > max_params:
      tas.check(mod.mods, msg=f'Cannot create module of size {n}: max_params={max_params}')

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

  def __init__(self, idim, odim, msize, tmgr, act=None):
    mdim = max(idim, odim)
    icount = (idim + msize - 1) // msize
    mcount = (mdim + msize - 1) // msize
    ocount = (odim + msize - 1) // msize

    super().__init__()
    self.odim = odim
    self.msize = msize
    self.fcin = nn.Linear(icount, mcount)
    self.mods = nn.ModuleList([tmgr.get(msize) for _ in range(mcount)])
    self.fcout = nn.Linear(mcount, ocount)
    self.act = act or nn.Identity()

  def forward(self, x):
    dims = list(x.shape)
    idim = dims.pop()

    rem = idim % self.msize
    if rem != 0:
      x = F.pad(x, (0, self.msize - rem), value=0)
      idim += self.msize - rem

    icount = idim // self.msize
    x = torch.reshape(x, (*dims, icount, self.msize)) # (*DIMS, ICOUNT, MSIZE)
    x = ut.tail_permute(x) # (*DIMS, MSIZE, ICOUNT)

    x = self.fcin(x) # (*DIMS, MSIZE, MCOUNT)

    x = ut.tail_permute(x) # (*DIMS, MCOUNT, MSIZE)

    parts = []
    for i, mod in enumerate(self.mods):
      yv = mod(torch.squeeze(x[:, i, :], 1))
      parts.append(torch.unsqueeze(yv, 1))

    xt = torch.cat(parts, dim=1) # (*DIMS, MCOUNT, MSIZE)
    x = ut.tail_permute(xt) # (*DIMS, MSIZE, MCOUNT)

    x = self.fcout(x) # (*DIMS, MSIZE, OCOUNT)

    x = torch.reshape(x, (*dims, -1))
    x = x[:, : self.odim] # (*DIMS, ODIM)

    x = self.act(x)

    return x

