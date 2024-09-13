import math

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

import py_misc_utils.assert_checks as tas


class Attention(nn.Module):

  def __init__(self, n_embd, n_head,
               attn_dropout=None,
               dropout=None):
    attn_dropout = attn_dropout or 0.0
    dropout = dropout or 0.0

    tas.check_eq(n_embd % n_head, 0,
                 msg=f'Embedding dimension ({n_embd}) must be multiple ' \
                 f'of the number of heads ({n_head})')

    super().__init__()
    self.n_head = n_head
    self.k_prj = nn.Linear(n_embd, n_embd * n_head, bias=False)
    self.q_prj = nn.Linear(n_embd, n_embd * n_head, bias=False)
    self.v_prj = nn.Linear(n_embd, n_embd * n_head, bias=False)
    self.unifyheads = nn.Linear(n_embd * n_head, n_embd)

    self.attn_drop = nn.Dropout(attn_dropout)
    self.resid_drop = nn.Dropout(dropout)

  def forward(self, k, q, v, mask=None):
    keys = einops.rearrange(self.k_prj(k), 'b t (h ch) -> b h t ch', h=self.n_head)
    queries = einops.rearrange(self.q_prj(q), 'b t (h ch) -> b h t ch', h=self.n_head)
    values = einops.rearrange(self.v_prj(v), 'b t (h ch) -> b h t ch', h=self.n_head)

    # (B, H, T, CH) @ (B, H, CH, T) => (B, H, T, T)
    att = queries @ einops.rearrange(keys, 'b h t ch -> b h ch t')
    if mask is not None:
      att = att.masked_fill(mask, float('-inf'))
    att = F.softmax(att / math.sqrt(queries.shape[-1]), dim=-1)
    att = self.attn_drop(att)

    # (B, H, T, T) @ (B, H, T, CH) => (B, H, T, CH)
    out = att @ values

    out = einops.rearrange(out, 'b h t ch -> b t (h ch)')
    # (B, T, C) @ (C, C) => (B, T, C)
    out = self.unifyheads(out)
    out = self.resid_drop(out)

    return out

