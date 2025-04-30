import math
import os

import einops
import py_misc_utils.alog as alog
import py_misc_utils.assert_checks as tas
import py_misc_utils.utils as pyu
import torch
import torch.nn as nn
import torch.nn.attention as nnatn

from . import core_utils as cu


class Attention(nn.Module):

  def __init__(self, embed_size, num_heads,
               bias=True,
               add_bias_kv=False,
               dropout=0.0):
    tas.check_eq(embed_size % num_heads, 0,
                 msg=f'The embed size ({embed_size}) should divide evenly by ' \
                 f'the number of heads ({num_heads})')

    super().__init__()
    self._num_heads = num_heads
    self.k_prj = nn.Linear(embed_size, embed_size, bias=add_bias_kv)
    self.q_prj = nn.Linear(embed_size, embed_size, bias=False)
    self.v_prj = nn.Linear(embed_size, embed_size, bias=add_bias_kv)
    self.attend = nn.Softmax(dim=-1)
    self.out_prj = nn.Linear(embed_size, embed_size, bias=bias)

    self.dropout = nn.Dropout(dropout)

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
    att = self.dropout(att)

    # (B, H, T, T) @ (B, H, T, N) => (B, H, T, N)
    out = att @ values

    out = einops.rearrange(out, 'b h t n -> b t (h n)')
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


_SDPA_KERNELS = {
  'MATH': nnatn.SDPBackend.MATH,
  'FLASH_ATTENTION': nnatn.SDPBackend.FLASH_ATTENTION,
  'EFFICIENT_ATTENTION': nnatn.SDPBackend.EFFICIENT_ATTENTION,
}
_SDPA_ALGO = os.getenv('SDPA_ALGO')

def raw_attention(q, k, v, mask=None):
  if _SDPA_ALGO is not None:
    with nnatn.sdpa_kernel(_SDPA_KERNELS[_SDPA_ALGO]):
      return nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
  else:
    return nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)

