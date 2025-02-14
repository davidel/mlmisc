import math

import einops
import py_misc_utils.alog as alog
import torch
import torch.nn as nn

from . import core_utils as cu


class SplitLinear(nn.Module):

  def __init__(self, in_features, out_features, compression=0.1):
    split_features = math.ceil(out_features * compression)
    num_parts = math.ceil(out_features / split_features)

    alog.debug(f'SplitFeatures: {split_features}, NumParts: {num_parts}')
    alog.debug(f'TotFeatures: {num_parts * split_features} vs. {out_features}')

    super().__init__()
    self.out_features = out_features
    self.num_parts = num_parts
    self.selector = nn.Linear(in_features, num_parts)
    self.expander = nn.Linear(num_parts, in_features)
    self.splitfc = nn.Linear(in_features, split_features)

  def forward(self, x):
    # (..., IN_FEAT) => (..., NPARTS, IN_FEAT)
    bx = einops.repeat(x, '... inf -> ... n inf', n=self.num_parts)

    # (..., IN_FEAT) => (..., NPARTS)
    sx = self.selector(x)

    # (..., NPARTS) => (..., NPARTS, NPARTS)
    sx = torch.diag_embed(sx)

    # (..., NPARTS, NPARTS) => (..., NPARTS, IN_FEAT)
    px = self.expander(sx)

    # (..., NPARTS, IN_FEAT) + (..., NPARTS, IN_FEAT) => (..., NPARTS, IN_FEAT)
    xx = bx + px

    # (..., NPARTS, IN_FEAT) => (..., NPARTS, SPLIT_FEAT)
    y = self.splitfc(xx)

    # (..., NPARTS, SPLIT_FEAT) => (..., NPARTS * SPLIT_FEAT)
    y = einops.rearrange(y, '... n sf -> ... (n sf)')

    return y[..., : self.out_features]

  def extra_repr(self):
    return cu.extra_repr(num_parts=self.num_parts,
                         in_features=self.splitfc.weight.shape[1],
                         split_features=self.splitfc.weight.shape[0],
                         tot_features=self.num_parts * self.splitfc.weight.shape[0],
                         out_features=self.out_features)

