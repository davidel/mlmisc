import torch
import torch.nn as nn

import py_misc_utils.assert_checks as tas

from . import layer_utils as lu


class Projection(nn.Module):

  def __init__(self, context_size, embed_size, embed_heads, vocab_size,
               fc_bias=True,
               act1=None,
               act2=None):
    tas.check_eq(embed_size % num_heads, 0,
                 f'Number of heads ({num_heads}) should divide the ' \
                 f'embedding size ({embed_size})')

    embed_k = embed_size // embed_heads
    prj_q = int(vocab_size // embed_heads)
    act1 = act1 or nn.Identity
    act2 = act2 or nn.Identity

    super().__init__()
    self.embed_heads = embed_heads
    self.prj1 = nn.Linear(context_size * embed_k, prj_q, bias=fc_bias)
    self.act1 = lu.create(act1)
    self.prj2 = nn.Linear(embed_heads * prj_q, vocab_size, bias=fc_bias)
    self.act2 = lu.create(act2)

  def forward(self, x):
    b, t, c = x.shape

    # (B, T, C) => (B, T, CH, CK)
    y = x.view(b, t, self.embed_heads, -1)
    # (B, T, CH, CK) => (B, CH, T, CK)
    y = torch.permute(y, (0, 2, 1, 3))
    # (B, CH, T, CK) => (B, CH, T * CK)
    y = y.reshape(b, self.embed_heads, -1)
    # (B, CH, T * CK) @ (T * CK, Q) => (B, CH, Q)
    y = self.act1(self.prj1(y))
    # (B, CH, Q) => (B, CH * Q)
    y = y.view(b, -1)
    # (B, CH * Q) @ (CH * Q, V) => (B, V)
    y = self.act2(self.prj2(y))

    return y

