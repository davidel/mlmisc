import torch
import torch.nn as nn

from . import layer_utils as lu


class EncoderBlock(nn.Module):

  def __init__(self, input_dim, num_heads,
               dim_feedforward=None,
               attn_dropout=None,
               dropout=None,
               act=None):
    dim_feedforward = dim_feedforward or (4 * input_dim)
    attn_dropout = attn_dropout or 0.0
    dropout = dropout or 0.0
    act = act or 'relu'

    super().__init__()
    self.attn = nn.MultiheadAttention(input_dim, num_heads,
                                      dropout=attn_dropout,
                                      batch_first=True)

    self.linear_net = nn.Sequential(
      nn.Linear(input_dim, dim_feedforward),
      nn.Dropout(dropout),
      lu.create(act),
      nn.Linear(dim_feedforward, input_dim)
    )

    self.norm1 = nn.LayerNorm(input_dim)
    self.norm2 = nn.LayerNorm(input_dim)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, mask=None):
    attn_out = self.attn(x, x, x, attn_mask=mask, need_weights=False)[0]
    x = x + self.dropout(attn_out)
    x = self.norm1(x)

    linear_out = self.linear_net(x)
    x = x + self.dropout(linear_out)
    x = self.norm2(x)

    return x

