import math
import os

import einops
import py_misc_utils.alog as alog
import py_misc_utils.assert_checks as tas
import py_misc_utils.utils as pyu
import torch
import torch.nn as nn

from . import core_utils as cu


def naive_attention(queries, keys, values, mask=None):
  att = queries @ torch.transpose(keys, -1, -2)
  if mask is not None:
    att = att.masked_fill(mask, float('-inf'))
  att = nn.functional.softmax(att / math.sqrt(queries.shape[-1]), dim=-1)
  out = att @ values

  return out


class Attention(nn.Module):

  def __init__(self, embed_size, num_heads,
               bias=True,
               add_bias_kv=False,
               dropout=0.0):
    tas.check_eq(embed_size % num_heads, 0,
                 msg=f'The embed size ({embed_size}) should divide evenly by ' \
                 f'the number of heads ({num_heads})')

    super().__init__()
    self.dropout = dropout if dropout != 0.0 else None
    self.k_prj = nn.Linear(embed_size, embed_size, bias=add_bias_kv)
    self.q_prj = nn.Linear(embed_size, embed_size, bias=False)
    self.v_prj = nn.Linear(embed_size, embed_size, bias=add_bias_kv)
    self.dropout = nn.Dropout(dropout)
    self.out_prj = nn.Linear(embed_size, embed_size, bias=bias)

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

    att = naive_attention(queries, keys, values, mask=mask)

    out = einops.rearrange(att, 'b h t n -> b t (h n)')
    out = self.dropout(out)
    # (B, T, C) @ (C, C) => (B, T, C)
    out = self.out_prj(out)

    return out

  def extra_repr(self):
    return cu.extra_repr(num_heads=self._num_heads,
                         embed_size=self.out_prj.out_features,
                         dropout=self.dropout.p)


class TorchAttention(nn.MultiheadAttention):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def forward(self, *args, mask=None, **kwargs):
    kwargs.update(
      need_weights=False,
      attn_mask=mask,
      is_causal=True if mask is not None else False,
    )

    y = super().forward(*args, **kwargs)

    return y[0]


class SelfAttention(nn.Module):

  def __init__(self, attn, **kwargs):
    super().__init__()
    self.attn = attn
    self.kwargs = kwargs

  def forward(self, x, **kwargs):
    attn_kwargs = self.kwargs.copy()
    attn_kwargs.update(kwargs)

    return self.attn(x, x, x, **attn_kwargs)


def create(embed_size, num_heads,
           attn_kind=None,
           bias=True,
           add_bias_kv=False,
           dropout=0.0,
           is_self=False):
  attn_kwargs = dict()

  if attn_kind in (None, 'torch_mha'):
    attn = TorchAttention(embed_size, num_heads,
                          bias=bias,
                          add_bias_kv=add_bias_kv,
                          dropout=dropout,
                          batch_first=True)
  elif attn_kind == 'internal':
    attn = Attention(embed_size, num_heads,
                     bias=bias,
                     add_bias_kv=add_bias_kv,
                     dropout=dropout)
  else:
    alog.xraise(ValueError, f'Invalid attention type: "{attn_kind}"')

  return SelfAttention(attn, **attn_kwargs) if is_self else attn


def raw_attention(q, k, v, mask=None):
  # att = naive_attention(q, k, v, mask=mask)
  return nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)

