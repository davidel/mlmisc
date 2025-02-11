import math

import py_misc_utils.alog as alog
import torch
import torch.nn as nn


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
    parts = []
    for i in range(self.num_parts):
      px = self.indexer(self.pos_embedding[i]) + x
      parts.append(self.splitfc(px))

    y = torch.cat(parts, dim=-1)

    return y[..., : self.out_features]

