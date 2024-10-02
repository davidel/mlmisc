import math

import einops
import torch
import torch.nn as nn

import py_misc_utils.assert_checks as tas
import py_misc_utils.utils as pyu

from . import layer_utils as lu


class TiledLinear(nn.Module):

  def __init__(self, in_features, out_features, num_tiles,
               bias=None,
               act=None):
    bias = True if bias is None else bias
    act = act or nn.Identity

    tile_size = math.ceil(in_features / num_tiles)

    super().__init__()
    self.num_tiles = num_tiles
    self.pad = num_tiles * tile_size - in_features
    self.tiled_fc = nn.Linear(tile_size, out_features, bias=bias)
    self.tiles_merger = nn.Linear(num_tiles, 1, bias=bias)
    self.act = lu.create(act)

  def forward(self, x):
    if self.pad:
      lpad = self.pad // 2
      rpad = self.pad - lpad
      y = nn.functional.pad(x, (lpad, rpad))
    else:
      y = x

    y = einops.rearrange(y, '... (nt ts) -> ... nt ts', nt=self.num_tiles)
    # ... nt ts -> ... nt out
    y = self.tiled_fc(y)
    y = einops.rearrange(y, '... nt out -> ... out nt')
    # ... out nt -> ... out 1
    y = self.tiles_merger(y)
    # ... out 1 -> ... out
    y = y.squeeze(-1)
    y = self.act(y)

    return y

  def extra_repr(self):
    return pyu.stri(dict(num_tiles=self.num_tiles,
                         tile_size=self.tiled_fc.weight.shape[-1],
                         pad=self.pad))

