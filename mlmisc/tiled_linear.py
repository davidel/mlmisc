import math

import einops
import py_misc_utils.utils as pyu
import torch
import torch.nn as nn

from . import utils as ut


class TiledLinear(nn.Module):

  def __init__(self, in_features, out_features, num_tiles,
               bias=True,
               crossed=False):
    tile_size = math.ceil(in_features / num_tiles)

    super().__init__()
    self.crossed = crossed
    self.num_tiles = num_tiles
    self.pad = num_tiles * tile_size - in_features
    self.tiled_fc = nn.Linear(tile_size, out_features, bias=bias)
    self.tiles_merger = nn.Linear(num_tiles, 1, bias=False)

  def forward(self, x):
    if self.pad:
      lpad = self.pad // 2
      rpad = self.pad - lpad
      y = nn.functional.pad(x, (lpad, rpad))
    else:
      y = x

    if self.crossed:
      y = einops.rearrange(y, '... (ts nt) -> ... nt ts', nt=self.num_tiles)
    else:
      y = einops.rearrange(y, '... (nt ts) -> ... nt ts', nt=self.num_tiles)
    # ... nt ts -> ... nt out
    y = self.tiled_fc(y)
    y = einops.rearrange(y, '... nt out -> ... out nt')
    # ... out nt -> ... out 1
    y = self.tiles_merger(y)
    # ... out 1 -> ... out
    y = y.squeeze(-1)

    return y

  def extra_repr(self):
    return ut.extra_repr(crossed=self.crossed,
                         num_tiles=self.num_tiles,
                         tile_size=self.tiled_fc.weight.shape[-1],
                         pad=self.pad)

