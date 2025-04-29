import math

import einops
import py_misc_utils.assert_checks as tas
import py_misc_utils.utils as pyu
import torch
import torch.nn as nn

from . import core_utils as cu
from . import layer_utils as lu


class ShardAttention(nn.Module):

  def __init__(self, embed_size, num_heads,
               post='id',
               post_feed=lambda x, y: y):
    tas.check_eq(embed_size % num_heads, 0,
                 f'Number of heads ({num_heads}) should divide the ' \
                 f'embedding size ({embed_size})')

    super().__init__()
    self.num_heads = num_heads
    self.weight = cu.kuni_parameter(num_heads * embed_size, embed_size)
    self.post = lu.create(post)
    self.post_feed = post_feed

  def forward(self, x, mask=None):
    q = k = einops.rearrange(x, 'b t (nh hs) -> b nh t hs', nh=self.num_heads)
    v = einops.repeat(x, 'b t c -> b nh t c', nh=self.num_heads)
    y = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    y = einops.rearrange(y, 'b nh t c -> b t (nh c)')
    y = y @ weight

    return self.post(self.post_feed(x, y))

  def extra_repr(self):
    return cu.extra_repr(num_heads=self.num_heads,
                         embed_size=self.weight.shape[-1])

