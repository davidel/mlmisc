import math

import einops
import torch
import torch.nn as nn


class Attention(nn.Module):

  def __init__(self, n_embd, n_head,
               attn_dropout=None,
               dropout=None):
    attn_dropout = attn_dropout or 0.0
    dropout = dropout or 0.0

    super().__init__()
    self.n_head = n_head
    self.k_prj = nn.Linear(n_embd, n_embd * n_head, bias=False)
    self.q_prj = nn.Linear(n_embd, n_embd * n_head, bias=False)
    self.v_prj = nn.Linear(n_embd, n_embd * n_head, bias=False)
    self.attend = nn.Softmax(dim=-1)
    self.unifyheads = nn.Linear(n_embd * n_head, n_embd)

    self.attn_drop = nn.Dropout(attn_dropout, inplace=True)
    self.resid_drop = nn.Dropout(dropout, inplace=True)

  def forward(self, k, q, v, mask=None):
    # (B, T, C) -> (B, T, H * C)
    keys = self.k_prj(k)
    # (B, T, C) -> (B, T, H * C)
    queries = self.q_prj(q)
    # (B, T, C) -> (B, T, H * C)
    values = self.v_prj(v)

    keys = einops.rearrange(keys, 'b t (h c) -> b h t c', h=self.n_head)
    queries = einops.rearrange(queries, 'b t (h c) -> b h t c', h=self.n_head)
    values = einops.rearrange(values, 'b t (h c) -> b h t c', h=self.n_head)

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

