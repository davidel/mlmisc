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


def norm_shape(x):
  if x.ndim == 3:
    return torch.unsqueeze(x, 0)
  if x.ndim > 4:
    return x.reshape(-1, *x.shape[-3:])

  return x


def generate_unfold(x, patch_specs):
  x = norm_shape(x)

  patches = []
  for ps in patch_specs:
    hpad = compute_pad(x.shape[-2], ps.hsize, ps.hstride, ps.hbase)
    wpad = compute_pad(x.shape[-1], ps.wsize, ps.wstride, ps.wbase)

    xpad = nn.functional.pad(x, wpad + hpad)
    xpatches = xpad.unfold(2, ps.hsize, ps.hstride).unfold(3, ps.wsize, ps.wstride)
    xpatches = einops.rearrange(xpatches, 'b c nh nw sh sw -> b (nh nw) (c sh sw)')

    patches.append(xpatches)

  return torch.cat(patches, dim=1)


def verify_patch_specs(patch_specs):
  patch_size = None
  for ps in patch_specs:
    size = ps.hsize * ps.wsize
    if patch_size is None:
      patch_size = size
    else:
      tas.check_eq(patch_size, size,
                   msg=f'All patches must have the same size (in terms of the ' \
                   f'HxW product): {patch_specs[0]} vs {ps}')


def create_convs(patch_specs, in_channels):
  convs = []
  for ps in patch_specs:
    convs.append(nn.Conv2d(in_channels, ps.hsize * ps.wsize * in_channels, (ps.hsize, ps.wsize),
                           stride=(ps.hsize, ps.wsize),
                           padding='valid'))

  return convs


def generate_conv(x, patch_specs, convs):
  x = norm_shape(x)

  patches = []
  for ps, conv in zip(patch_specs, convs):
    hpad = compute_pad(x.shape[-2], ps.hsize, ps.hstride, ps.hbase)
    wpad = compute_pad(x.shape[-1], ps.wsize, ps.wstride, ps.wbase)

    xpad = nn.functional.pad(x, wpad + hpad)
    xpatches = conv(xpad)
    xpatches = einops.rearrange(xpatches, 'b c h w -> b (h w) c')

    patches.append(xpatches)

  return torch.cat(patches, dim=1)


CONV = 'conv'
UNFOLD = 'unfold'
PATCH_MODES = {CONV, UNFOLD}

class Patcher(nn.Module):

  def __init__(self, patch_specs, mode=CONV, in_channels=None):
    verify_patch_specs(patch_specs)
    tas.check(mode in PATCH_MODES,
              msg=f'Unknown patcher mode (should be one of {PATCH_MODES}): {mode}')
    tas.check(mode != CONV or in_channels is not None,
              msg=f'The in_channels argument must be specified in conv mode')

    super().__init__()
    self.patch_specs = patch_specs
    self.mode = mode
    if mode == CONV:
      self.convs = nn.ModuleList(create_convs(patch_specs, in_channels))

  def forward(self, x):
    if self.mode == UNFOLD:
      y = generate_unfold(x, self.patch_specs)
    else:
      y = generate_conv(x, self.patch_specs, self.convs)

    return y

