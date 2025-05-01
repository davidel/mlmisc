import math
import os

import einops
import py_misc_utils.alog as alog
import py_misc_utils.assert_checks as tas
import py_misc_utils.utils as pyu
import torch
import torch.autograd as agrad
import torch.nn as nn

from . import core_utils as cu


class AttentionFunction(agrad.function.Function):

  @staticmethod
  def forward(ctx, q, k, v, mask, tile_size, split_count):
    batch, *extra, context_size, embed_size = q.shape

    tile_size = tile_size or math.ceil(context_size / split_count)
    scale = 1.0 / math.sqrt(embed_size)

    q_tiles = q.split(tile_size, dim=-2)
    k_tiles = k.split(tile_size, dim=-2)
    v_tiles = v.split(tile_size, dim=-2)

    smax_sum = torch.zeros(q.shape[: -1], dtype=q.dtype, device=q.device)
    smax_max = torch.full_like(smax_sum, float('-inf'))

    smax_sum_tiles = smax_sum.split(tile_size, dim=-1)
    smax_max_tiles = smax_max.split(tile_size, dim=-1)

    if mask is not None:
      xmask = mask.reshape((1,) * (1 + len(extra)) + tuple(mask.shape))
      masks = [torch.split(m, tile_size, dim=-1) for m in torch.split(xmask, tile_size, dim=-2)]
    else:
      masks = None

    out_tiles = []
    for iq, q_tile in enumerate(q_tiles):
      out_tile = torch.zeros_like(v_tiles[iq])
      smax_sum_tile = smax_sum_tiles[iq]
      smax_max_tile = smax_max_tiles[iq]

      for ik, k_tile in enumerate(k_tiles):
        qk = q_tile @ torch.transpose(k_tile, -1, -2)

        qk.mul_(scale)
        if masks is not None:
          qk.masked_fill_(masks[iq][ik], float('-inf'))

        curr_max = torch.max(qk, dim=-1).values
        curr_max = torch.maximum(curr_max, smax_max_tile)
        scaler = torch.exp(smax_max_tile - curr_max)

        qk.sub_(curr_max.unsqueeze(-1))
        qk.exp_()

        smax_sum_tile.mul_(scaler)
        smax_sum_tile.add_(torch.sum(qk, dim=-1))

        qkv = qk @ v_tiles[ik]

        out_tile.mul_(scaler.unsqueeze(-1))
        out_tile.add_(qkv)

        smax_max_tile.copy_(curr_max)

      out_tiles.append(out_tile)

    out = torch.cat(out_tiles, dim=-2)

    smax_sum = smax_sum.reshape(tuple(smax_sum.shape) + (1,) * (out.ndim - smax_sum.ndim))
    smax_max = smax_max.reshape(tuple(smax_max.shape) + (1,) * (out.ndim - smax_max.ndim))

    out.div_(smax_sum)

    exp_scaler = smax_sum.log() + smax_max

    ctx.args = (scale, masks, tile_size)
    ctx.save_for_backward(q, k, v, out, exp_scaler)

    return out

  @staticmethod
  def backward(ctx, dout):
    scale, masks, tile_size = ctx.args
    q, k, v, out, exp_scaler = ctx.saved_tensors

    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)

    q_tiles = q.split(tile_size, dim=-2)
    k_tiles = k.split(tile_size, dim=-2)
    v_tiles = v.split(tile_size, dim=-2)

    dq_tiles = dq.split(tile_size, dim=-2)
    dk_tiles = dk.split(tile_size, dim=-2)
    dv_tiles = dv.split(tile_size, dim=-2)

    out_tiles = out.split(tile_size, dim=-2)
    exp_scaler_tiles = exp_scaler.split(tile_size, dim=-2)
    dout_tiles = dout.split(tile_size, dim=-2)

    for iq, q_tile in enumerate(q_tiles):
      exp_scaler_tile = exp_scaler_tiles[iq]
      dout_tile = dout_tiles[iq]
      out_tile = out_tiles[iq]
      dq_tile = dq_tiles[iq]

      for ik, k_tile in enumerate(k_tiles):
        dk_tile = dk_tiles[ik]
        dv_tile = dv_tiles[ik]
        v_tile = v_tiles[ik]

        qk = q_tile @ torch.transpose(k_tile, -1, -2)

        qk.mul_(scale)
        if masks is not None:
          qk.masked_fill_(masks[iq][ik], float('-inf'))

        qk.sub_(exp_scaler_tile)
        qk.exp_()

        dv_grad = torch.transpose(qk, -1, -2) @ dout_tile
        v_grad = dout_tile @ torch.transpose(v_tile, -1, -2)

        o_grad = (dout_tile * out_tile).sum(dim=-1, keepdims=True)
        ds = qk * scale * (v_grad - o_grad)

        dq_grad = ds @ k_tile
        dk_grad = torch.transpose(ds, -1, -2) @ q_tile

        dq_tile.add_(dq_grad)
        dk_tile.add_(dk_grad)
        dv_tile.add_(dv_grad)

    return dq, dk, dv, None, None, None


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

    att = raw_attention(queries, keys, values, mask=mask)

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


def naive_attention(queries, keys, values, mask=None):
  att = queries @ torch.transpose(keys, -1, -2)
  if mask is not None:
    att = att.masked_fill(mask, float('-inf'))
  att = nn.functional.softmax(att / math.sqrt(queries.shape[-1]), dim=-1)
  out = att @ values

  return out


_ATTENTION_MODE = os.getenv('ATTENTION_MODE')

def raw_attention(queries, keys, values, mask=None):
  if _ATTENTION_MODE in (None, 'naive'):
    return naive_attention(queries, keys, values, mask=mask)
  elif _ATTENTION_MODE == 'flash':
    return AttentionFunction.apply(queries, keys, values, mask, 512, None)
  elif _ATTENTION_MODE == 'sdp':
    # In theory, this should be used ... but it is a NaN generator at the time of
    # writing, so I use the naive one for now.
    return nn.functional.scaled_dot_product_attention(queries, keys, values,
                                                      attn_mask=mask)
  else:
    alog.xraise(ValueError, f'Invalid attention mode: "{_ATTENTION_MODE}"')

