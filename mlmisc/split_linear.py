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

    indexer_features = max(16, in_features // 8)
    pos_embedding = torch.linspace(0.0, 1.0, num_parts, dtype=torch.float32)

    super().__init__()
    self.out_features = out_features
    self.num_parts = num_parts
    self.indexer = nn.Sequential(
      nn.Linear(1, indexer_features),
      nn.ReLU(),
      nn.Linear(indexer_features, in_features),
      nn.ReLU(),
    )
    self.splitfc = nn.Linear(in_features, split_features)
    self.register_buffer('pos_embedding', pos_embedding.reshape((num_parts, 1)))

  def forward(self, x):
    # (NPARTS, 1) @ (1, IN_FEAT) => (NPARTS, IN_FEAT)
    ex = self.indexer(self.pos_embedding)

    # (..., IN_FEAT) -> (..., NPARTS, IN_FEAT)
    bx = cu.add_dimension(x, -2, self.num_parts)

    # (..., NPARTS, IN_FEAT) + (NPARTS, IN_FEAT) => (..., NPARTS, IN_FEAT)
    xx = bx + ex

    # (..., NPARTS, IN_FEAT) @ (IN_FEAT, SPLIT_FEAT) => (..., NPARTS, SPLIT_FEAT)
    y = self.splitfc(xx)

    # (..., NPARTS, SPLIT_FEAT) => (..., NPARTS * SPLIT_FEAT)
    y = einops.rearrange(y, '... n sf -> ... (n sf)')

    return y[..., : self.out_features]

