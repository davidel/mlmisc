import math

import einops
import py_misc_utils.assert_checks as tas
import torch
import torch.nn as nn

from . import layer_utils as lu
from . import utils as ut


class ShardAttention(nn.Module):

  def __init__(self, embed_size, num_heads, post=None, post_feed=None):
    tas.check_eq(embed_size % num_heads, 0,
                 f'Number of heads ({num_heads}) should divide the ' \
                 f'embedding size ({embed_size})')

    super().__init__()
    self.num_heads = num_heads
    self.weight = nn.Parameter(ut.kuni_tensor(num_heads * embed_size, embed_size))
    self.attend = nn.Softmax(dim=-1)
    self.post = lu.create(post or nn.Identity)
    self.post_feed = (lambda x, y: y) if post_feed is None else post_feed

  def forward(self, x, mask=None):
    y = einops.rearrange(x, 'b t (ch ck) -> b ch t ck', ch=self.num_heads)
    yt = einops.rearrange(y, 'b ch t ck -> b ch ck t')
    # (B, CH, T, CK) @ (B, CH, CK, T) => (B, CH, T, T)
    y = torch.einsum('bhtk,bhks->bhts', y, yt)

    if mask is not None:
      y = y.masked_fill(mask, float('-inf'))
    y = self.attend(y / math.sqrt(y.shape[1]))

    xx = einops.repeat(x, 'b t c -> b ch t c', ch=self.num_heads)
    # (B, CH, T, T) @ (B, CH, T, C) => (B, CH, T, C)
    y = torch.einsum('bhts,bhsk->bhtk', y, xx)
    y = einops.rearrange(y, 'b ch t c -> b t (ch c)')
    # (B, T, CH * C) @ (CH * C, C) => (B, T, C)
    y = y @ self.weight

    return self.post(self.post_feed(x, y))

  def extra_repr(self):
    return pyu.stri(dict(num_heads=self.num_heads,
                         embed_size=self.weight.shape[-1]))

