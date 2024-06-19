import torch
import torch.nn as nn


class EncoderBlock(nn.Module):

  def __init__(self, input_dim, num_heads, dim_feedforward=None, dropout=None):
    super().__init__()

    dim_feedforward = dim_feedforward or (4 * input_dim)
    dropout = dropout or 0

    self.attn = nn.MultiheadAttention(input_dim, num_heads,
                                      dropout=dropout,
                                      batch_first=True)

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
    attn_out = self.attn(x, x, x, attn_mask=mask, need_weights=False)
    x = x + self.dropout(attn_out[0])
    x = self.norm1(x)

    linear_out = self.linear_net(x)
    x = x + self.dropout(linear_out)
    x = self.norm2(x)

    return x
