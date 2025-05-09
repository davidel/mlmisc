import math

import einops
import py_misc_utils.assert_checks as tas
import py_misc_utils.utils as pyu
import torch
import torch.nn as nn

from . import attention as atn
from . import core_utils as cu
from . import layer_utils as lu


class ShardAttention(nn.Module):

  def __init__(self, embed_size, num_heads):
    tas.check_eq(embed_size % num_heads, 0,
                 f'Number of heads ({num_heads}) should divide the ' \
                 f'embedding size ({embed_size})')

    super().__init__()
    self.num_heads = num_heads
    self.weight = cu.kuni_parameter(num_heads * embed_size, embed_size)

  def forward(self, x, mask=None):
    q = k = einops.rearrange(x, 'b t (nh hs) -> b nh t hs', nh=self.num_heads)
    v = einops.repeat(x, 'b t c -> b nh t c', nh=self.num_heads)
    y = atn.raw_attention(q, k, v, mask=mask)
    y = einops.rearrange(y, 'b nh t c -> b t (nh c)')
    y = y @ self.weight

    return y

  def extra_repr(self):
    return cu.extra_repr(num_heads=self.num_heads,
                         embed_size=self.weight.shape[-1])

