import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import py_misc_utils.assert_checks as tas

from . import layer_utils as lu


class ShardAttention(nn.Module):

  def __init__(self, embed_size, num_heads, post=None, post_feed=None):
    tas.check_eq(embed_size % num_heads, 0,
                 f'Number of heads ({num_heads}) should divide the ' \
                 f'embedding size ({embed_size})')

    super().__init__()
    self.num_heads = num_heads
    self.weight = nn.Parameter(torch.randn(num_heads * embed_size, embed_size))
    self.post = lu.create(post or nn.Identity)
    self.post_feed = (lambda x, y: y) if post_feed is None else post_feed

  def forward(self, x, mask=None):
    b, t, c = x.shape

    # (B, T, C) => (B, T, CH, CK)
    y = x.view(b, t, self.num_heads, -1)
    # (B, T, CH, CK) => (B, CH, T, CK)
    y = torch.permute(y, (0, 2, 1, 3))
    # (B, CH, T, CK) => (B, CH, CK, T)
    yt = torch.permute(y, (0, 1, 3, 2))
    # (B, CH, T, CK) @ (B, CH, CK, T) => (B, CH, T, T)
    y = torch.einsum('bhtk,bhks->bhts', y, yt)

    if mask is not None:
      y = y.masked_fill(mask, float('-inf'))
    y = F.softmax(y / math.sqrt(y.shape[1]), dim=-1)

    # (B, T, C) => (B, 1, T, C)
    xx = x.unsqueeze(1)
    # (B, 1, T, C) => (B, CH, T, C)
    xx = xx.expand(b, self.num_heads, t, c)

    # (B, CH, T, T) @ (B, CH, T, C) => (B, CH, T, C)
    y = torch.einsum('bhts,bhsk->bhtk', y, xx)
    # (B, CH, T, C) => (B, T, CH, C)
    y = torch.permute(y, (0, 2, 1, 3))
    # (B, T, CH, C) => (B, T, CH * C)
    y = y.reshape(b, t, -1)

    # (B, T, CH * C) @ (CH * C, C) => (B, T, C)
    y = y @ self.weight

    return self.post(self.post_feed(x, y))

