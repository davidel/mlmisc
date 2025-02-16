import einops
import torch
import torch.nn as nn

from . import core_utils as cu


class CrossLinear(nn.Module):

  def __init__(self, context_size, embed_size, bias=True):
    super().__init__()
    self.context_size = context_size
    self.morpher = cu.kuni_parameter(embed_size, embed_size)

  def forward(self, x, mask=None):
    # (B, T, C) @ (B, C, T) => (B, T, T)
    y = x @ einops.rearrange(x, 'b t c -> b c t')

    if mask is not None:
      y = y.masked_fill(mask, float('-inf'))

    y = nn.functional.softmax(y, dim=-1)

    # (B, T, C) @ (C, C) => (B, T, C)
    xx = x @ self.morpher

    # (B, T, T) @ (B, T, C) => (B, T, C)
    y = y @ xx

    return y

  def extra_repr(self):
    return cu.extra_repr(context_size=self.context_size,
                         embed_size=self.morpher.shape[-1])

