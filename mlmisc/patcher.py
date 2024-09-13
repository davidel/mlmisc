import collections

import einops
import py_misc_utils.assert_checks as tas
import py_misc_utils.utils as pyu
import torch
import torch.nn as nn


Patch = collections.namedtuple('Patch', 'hsize, wsize, hstride, wstride, hbase, wbase',
                               defaults=(0, 0))


def compute_pad(size, wnd_size, stride, base):
  tas.check_lt(base, wnd_size,
               msg=f'Patch base must be lower than its size: {wnd_size} vs {base}')

  if base > 0:
    eff_size, lpad = size - base, wnd_size - base
  else:
    eff_size, lpad = size, 0

  rem = eff_size % stride
  if rem > 0:
    nsize = max(eff_size, eff_size - rem + wnd_size)
  else:
    nsize = max(eff_size, eff_size - stride + wnd_size)

  pad = nsize - eff_size
  if pad > 0:
    if lpad > 0:
      return lpad, pad
    else:
      rpad = pad // 2

      return pad - rpad, rpad

  return lpad, 0


def generate_patches(x, patch_specs):
  if x.ndim == 3:
    x = torch.unsqueeze(x, 0)
  elif x.ndim > 4:
    x = x.reshape(-1, *x.shape[-3: ])

  patches = []
  for ps in patch_specs:
    hpad = compute_pad(x.shape[-2], ps.hsize, ps.hstride, ps.hbase)
    wpad = compute_pad(x.shape[-1], ps.wsize, ps.wstride, ps.wbase)

    xpad = nn.functional.pad(x, wpad + hpad)
    xpatches = xpad.unfold(2, ps.hsize, ps.hstride).unfold(3, ps.wsize, ps.wstride)
    xpatches = einops.rearrange(xpatches, 'b c nh nw sh sw -> b (nh nw) (c sh sw)')

    patches.append(xpatches)

  return torch.cat(patches, dim=1)


class Patcher(nn.Module):

  def __init__(self, patches):
    super().__init__()
    self.patches = patches

  def forward(self, x):
    return generate_patches(x, self.patches)

