import math

import einops
import py_misc_utils.assert_checks as tas
import py_misc_utils.utils as pyu
import torch
import torch.nn as nn

from . import core_utils as cu


class Attention(nn.Module):

  def __init__(self, embed_size, num_heads,
               attn_dropout=0.0,
               dropout=0.0):
    tas.check_eq(embed_size % num_heads, 0,
                 msg=f'The embed size ({embed_size}) should divide evenly by ' \
                 f'the number of heads ({num_heads})')

    super().__init__()
    self._num_heads = num_heads
    self.k_prj = nn.Linear(embed_size, embed_size, bias=False)
    self.q_prj = nn.Linear(embed_size, embed_size, bias=False)
    self.v_prj = nn.Linear(embed_size, embed_size, bias=False)
    self.attend = nn.Softmax(dim=-1)
    self.out_prj = nn.Linear(embed_size, embed_size)

    self.attn_drop = nn.Dropout(attn_dropout)
    self.resid_drop = nn.Dropout(dropout)

  def forward(self, k, q, v, mask=None):
    # (B, T, C) -> (B, T, C)
    keys = self.k_prj(k)
    # (B, T, C) -> (B, T, C)
    queries = self.q_prj(q)
    # (B, T, C) -> (B, T, C)
    values = self.v_prj(v)

    keys = einops.rearrange(keys, 'b t (h n) -> b h t n', h=self._num_heads)
    queries = einops.rearrange(queries, 'b t (h n) -> b h t n', h=self._num_heads)
    values = einops.rearrange(values, 'b t (h n) -> b h t n', h=self._num_heads)

    # (B, H, T, N) @ (B, H, N, T) => (B, H, T, T)
    att = queries @ einops.rearrange(keys, 'b h t n -> b h n t')
    if mask is not None:
      att = att.masked_fill(mask, float('-inf'))
    att = self.attend(att / math.sqrt(queries.shape[-1]))
    att = self.attn_drop(att)

    # (B, H, T, T) @ (B, H, T, N) => (B, H, T, N)
    out = att @ values

    out = einops.rearrange(out, 'b h t n -> b t (h n)')
    # (B, T, C) @ (C, C) => (B, T, C)
    out = self.out_prj(out)
    out = self.resid_drop(out)

    return out

  def extra_repr(self):
    return cu.extra_repr(num_heads=self._num_heads,
                         embed_size=self.out_prj.out_features,
                         attn_dropout=self.attn_drop.p,
                         dropout=self.resid_drop.p)


class SelfAttention(Attention):

  def forward(self, x, **kwargs):
    return super().forward(x, x, x, **kwargs)

