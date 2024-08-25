import collections
import bisect
import math

import py_misc_utils.alog as alog
import py_misc_utils.assert_checks as tas
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import layer_utils as lu


class Tile(nn.Module):

  def __init__(self, n, dtype=None, init=None):
    super().__init__()
    weight = torch.empty(n, n, dtype=dtype)
    (init or nn.init.kaiming_uniform_)(weight)
    self.weight = nn.Parameter(weight)


class TilesPod(nn.Module):

  def __init__(self, msize, count, dtype=None, init=None):
    super().__init__()
    self.msize = msize
    self.count = count
    self.dtype = dtype
    self.init = init
    self.mods = nn.ModuleList()
    self.reset()

  def reset(self):
    self.idx = 0
    self.used = 0

  def get_tile(self):
    if len(self.mods) >= self.count:
      m = self.mods[self.idx]
      self.idx = (self.idx + 1) % len(self.mods)
    else:
      m = Tile(self.msize, dtype=self.dtype, init=self.init)
      self.mods.append(m)

    self.used += 1

    return m

  def state_dict(self, *args, **kwargs):
    state = super().state_dict(*args, **kwargs)
    with torch.no_grad():
      state['mods'] = torch.hstack([m.weight for m in self.mods])

    return state

  def load_state_dict(self, state, *args, **kwargs):
    missing, known_keys = [], ('mods',)
    stack = state.pop('mods', None)
    if stack is None:
      if kwargs.get('strict', True):
        alog.xraise(ValueError, f'Input state mossing "mods" key')
      missing.append('mods')
    else:
      assign = kwargs.get('assign', False)
      for i, m in enumerate(self.mods):
        param_window = stack[:, i * self.msize: (i + 1) * self.msize]
        if assign:
          new_param = nn.Parameter(param_window.detach().clone(),
                                   requires_grad=m.weight.requires_grad)
          torch.utils.swap_tensors(m.weight, new_param)
        else:
          m.weight.copy_(param_window)

    extra = list(set(state.keys()) - set(known_keys))
    alog.info(f'EXTRA = {extra}\tKEYS = {list(state.keys())}\tKNOWN = {list(known_keys)}')

    return missing, list(set(state.keys()) - set(known_keys))


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
      stats[n] = dict(count=len(mod.mods),
                      used=mod.used,
                      params=len(mod.mods) * n**2)

    return stats

  def build_modules(self, msize, icount, ocount):
    mod = self.get(msize)

    parts = []
    for i in range(icount):
      parts.append([mod.get_tile().weight for _ in range(ocount)])

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
      fc_mat = torch.vstack([torch.hstack(oparts) for oparts in self.parts])
    else:
      fc_mat = self.fc_mat
      if fc_mat is None:
        with torch.no_grad():
          self.fc_mat = fc_mat = torch.vstack([torch.hstack(oparts) for oparts in self.parts])

    return fc_mat

  def forward(self, x):
    fc_mat = self._get_fc_mat()

    x = self.pad(x)
    x = x @ fc_mat
    x = x[..., : self.odim]
    x = self.post(x + self.bias)

    return x

