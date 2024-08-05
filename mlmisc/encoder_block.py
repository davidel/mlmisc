import torch
import torch.nn as nn

from . import attention as atn
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
    act = act or nn.GELU

    super().__init__()
    self.attn = atn.Attention(input_dim, num_heads,
                              dropout=attn_dropout)

    self.linear_net = nn.Sequential(
      nn.Linear(input_dim, dim_feedforward),
      nn.Dropout(dropout),
      lu.create(act),
      nn.Linear(dim_feedforward, input_dim)
    )

    self.norm1 = nn.LayerNorm(input_dim)
    self.norm2 = nn.LayerNorm(input_dim)

  def forward(self, x, mask=None):
    xn = self.norm1(x)
    x = x + self.attn(xn, xn, xn, mask=mask)
    x = x + self.linear_net(self.norm2(x))

    return x

