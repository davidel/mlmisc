import math

import einops
import py_misc_utils.utils as pyu
import torch
import torch.nn as nn


class Attention(nn.Module):

  def __init__(self, embed_size, num_heads,
               attn_dropout=None,
               dropout=None):
    attn_dropout = pyu.value_or(attn_dropout, 0.0)
    dropout = pyu.value_or(dropout, 0.0)

    super().__init__()
    self.num_heads = num_heads
    self.k_prj = nn.Linear(embed_size, embed_size * num_heads, bias=False)
    self.q_prj = nn.Linear(embed_size, embed_size * num_heads, bias=False)
    self.v_prj = nn.Linear(embed_size, embed_size * num_heads, bias=False)
    self.attend = nn.Softmax(dim=-1)
    self.unifyheads = nn.Linear(embed_size * num_heads, embed_size)

    self.attn_drop = nn.Dropout(attn_dropout)
    self.resid_drop = nn.Dropout(dropout)

  def forward(self, k, q, v, mask=None):
    # (B, T, C) -> (B, T, H * C)
    keys = self.k_prj(k)
    # (B, T, C) -> (B, T, H * C)
    queries = self.q_prj(q)
    # (B, T, C) -> (B, T, H * C)
    values = self.v_prj(v)

    keys = einops.rearrange(keys, 'b t (h c) -> b h t c', h=self.num_heads)
    queries = einops.rearrange(queries, 'b t (h c) -> b h t c', h=self.num_heads)
    values = einops.rearrange(values, 'b t (h c) -> b h t c', h=self.num_heads)

    # (B, H, T, C) @ (B, H, C, T) => (B, H, T, T)
    att = queries @ einops.rearrange(keys, 'b h t c -> b h c t')
    if mask is not None:
      att = att.masked_fill(mask, float('-inf'))
    att = self.attend(att / math.sqrt(queries.shape[-1]))
    att = self.attn_drop(att)

    # (B, H, T, T) @ (B, H, T, C) => (B, H, T, C)
    out = att @ values

    out = einops.rearrange(out, 'b h t c -> b t (h c)')
    # (B, T, H * C) @ (H * C, C) => (B, T, C)
    out = self.unifyheads(out)
    out = self.resid_drop(out)

    return out

  def extra_repr(self):
    return pyu.stri(dict(num_heads=self.num_heads,
                         embed_size=self.unifyheads.out_features,
                         attn_dropout=self.attn_drop.p,
                         dropout=self.resid_drop.p))


class SelfAttention(Attention):

  def forward(self, x, **kwargs):
    return super().forward(x, x, x, **kwargs)

