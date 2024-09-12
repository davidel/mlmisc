import einops
import py_misc_utils.utils as pyu
import torch
import torch.nn as nn


def compute_pad(size, wnd_size, stride):
  rem = size % stride
  if rem > 0:
    nsize = size - rem + wnd_size
    pad = nsize - size
    if pad > 0:
      rpad = pad // 2

      return pad - rpad, rpad

  return 0, 0


def generate_patches(x, size, strides):
  if isinstance(size, (list, tuple)):
    hsize, wsize = size
  else:
    hsize, wsize = size, size

  if x.ndim == 3:
    x = torch.unsqueeze(x, 0)
  elif x.ndim > 4:
    x = x.reshape(-1, *x.shape[-3: ])

  patches = []
  for stride in pyu.as_sequence(strides):
    hpad = compute_pad(x.shape[-2], hsize, stride)
    wpad = compute_pad(x.shape[-1], wsize, stride)

    xpad = nn.functional.pad(x, wpad + hpad)

    xpatches = einops.rearrange(xpad, 'b c (nh sh) (nw sw) -> b (nh nw) (c sh sw)',
                                sh=hsize,
                                sw=wsize)
    patches.append(xpatches)

  return torch.cat(patches, dim=1)


class Patcher(nn.Module):

  def __init__(self, patch_size, strides):
    super().__init__()
    self.patch_size = patch_size
    self.strides = strides

  def forward(self, x):
    return generate_patches(x, self.patch_size, self.strides)

