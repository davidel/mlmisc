import math

import einops
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
    y = einops.rearrange(x, 'b t (ch ck) -> b ch t ck', ch=self.num_heads)
    yt = einops.rearrange(y, 'b ch t ck -> b ch ck t')
    # (B, CH, T, CK) @ (B, CH, CK, T) => (B, CH, T, T)
    y = torch.einsum('bhtk,bhks->bhts', y, yt)

    if mask is not None:
      y = y.masked_fill(mask, float('-inf'))
    y = F.softmax(y / math.sqrt(y.shape[1]), dim=-1)

    xx = einops.repeat(x, 'b t c -> b ch t c', ch=self.num_heads)
    # (B, CH, T, T) @ (B, CH, T, C) => (B, CH, T, C)
    y = torch.einsum('bhts,bhsk->bhtk', y, xx)
    y = einops.rearrange(y, 'b ch t c -> b t (ch c)')
    # (B, T, CH * C) @ (CH * C, C) => (B, T, C)
    y = y @ self.weight

    return self.post(self.post_feed(x, y))

