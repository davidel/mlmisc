import math

import torch
import torch.nn as nn

import py_misc_utils.assert_checks as tas


class Head(nn.Module):

  def __init__(self, embed_size, head_size, dropout=None, bias=False):
    super().__init__()
    self.key = nn.Linear(embed_size, head_size, bias=bias)
    self.query = nn.Linear(embed_size, head_size, bias=bias)
    self.value = nn.Linear(embed_size, head_size, bias=bias)
    self.dropout = nn.Dropout(dropout) if dropout is not None else nn.Identity()

  def forward(self, k, q, v, mask=None):
    B, T, C = k.shape

    xk = self.key(k)   # (B, T, C)
    xq = self.query(q) # (B, T, C)
    wei = xq @ xk.transpose(-2, -1) / math.sqrt(C) # (B, T, C) @ (B, C, T) -> (B, T, T)
    if mask is not None:
      wei = wei.masked_fill(mask, float('-inf')) # (B, T, T)
    wei = nn.functional.softmax(wei, dim=-1) # (B, T, T)
    wei = self.dropout(wei)
    xv = self.value(v) # (B, T, C)
    out = wei @ xv # (B, T, T) @ (B, T, C) -> (B, T, C)

    return out


class MultiHeadAttention(nn.Module):

  def __init__(self, embed_size, num_heads, dropout=None):
    super().__init__()
    tas.check_eq(embed_size % num_heads, 0,
                 msg=f'Embed size ({embed_size}) must be multiple of the number of heads ({num_heads})')
    head_size = embed_size // num_heads
    heads = [
      Head(embed_size, head_size, dropout=dropout) for _ in range(num_heads)
    ]
    self.heads = nn.ModuleList(heads)
    self.proj = nn.Linear(embed_size, embed_size)
    self.dropout = nn.Dropout(dropout) if dropout is not None else nn.Identity()

  def forward(self, k, q, v, mask=None):
    out = torch.cat([h(k, q, v, mask=mask) for h in self.heads], dim=-1)
    out = self.dropout(self.proj(out))

    return out


class EncoderBlock(nn.Module):

  def __init__(self, input_dim, num_heads,
               dim_feedforward=None,
               attn_dropout=None,
               dropout=None):
    super().__init__()

    dim_feedforward = dim_feedforward or (4 * input_dim)
    attn_dropout = attn_dropout or 0
    dropout = dropout or 0

    # self.attn = nn.MultiheadAttention(input_dim, num_heads,
    #                                   dropout=attn_dropout,
    #                                   batch_first=True)
    self.attn = MultiHeadAttention(input_dim, num_heads,
                                   dropout=attn_dropout)

    self.linear_net = nn.Sequential(
        nn.Linear(input_dim, dim_feedforward),
        nn.Dropout(dropout),
        nn.ReLU(inplace=True),
        nn.Linear(dim_feedforward, input_dim)
    )

    self.norm1 = nn.LayerNorm(input_dim)
    self.norm2 = nn.LayerNorm(input_dim)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, mask=None):
    # attn_out = self.attn(x, x, x, attn_mask=mask, need_weights=False)[0]
    attn_out = self.attn(x, x, x, mask=mask)
    x = x + self.dropout(attn_out)
    x = self.norm1(x)

    linear_out = self.linear_net(x)
    x = x + self.dropout(linear_out)
    x = self.norm2(x)

    return x
