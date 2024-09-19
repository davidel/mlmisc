import einops
import torch
import torch.nn as nn

import py_misc_utils.assert_checks as tas

from . import layer_utils as lu


class Projection(nn.Module):

  def __init__(self, context_size, embed_size, num_heads, vocab_size,
               fc_bias=True,
               act1=None,
               act2=None):
    tas.check_eq(embed_size % num_heads, 0,
                 f'Number of heads ({num_heads}) should divide the ' \
                 f'embedding size ({embed_size})')

    embed_k = embed_size // num_heads
    prj_q = int(vocab_size // num_heads)
    act1 = act1 or nn.Identity
    act2 = act2 or nn.Identity

    super().__init__()
    self.num_heads = num_heads
    self.prj1 = nn.Linear(context_size * embed_k, prj_q, bias=fc_bias)
    self.act1 = lu.create(act1)
    self.prj2 = nn.Linear(num_heads * prj_q, vocab_size, bias=fc_bias)
    self.act2 = lu.create(act2)

  def forward(self, x):
    y = einops.rearrange(x, 'b t (h c) -> b h (t c)', h=self.num_heads)
    # (B, H, T * C) @ (T * C, Q) => (B, H, Q)
    y = self.act1(self.prj1(y))
    y = einops.rearrange(y, 'b h q -> b (h q)')
    # (B, H * Q) @ (H * Q, V) => (B, V)
    y = self.act2(self.prj2(y))

    return y

