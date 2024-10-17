import math

import einops
import py_misc_utils.utils as pyu
import torch
import torch.nn as nn

from . import layer_utils as lu
from .models.sequence import sequence_base as sb


class ForkAttention(nn.Module):

  def __init__(self, context_size, embed_size,
               post=None,
               post_feed=None):
    post = pyu.value_or(post, nn.Identity)

    super().__init__()
    self.fc = nn.Linear(embed_size, embed_size)
    self.alt_fc = nn.Linear(context_size, context_size)
    self.vfc = nn.Linear(embed_size, embed_size)
    self.attend = nn.Softmax(dim=-1)
    self.post = lu.create(post)
    self.post_feed = pyu.value_or(post_feed, lambda x, y: y)

  def forward(self, x, mask=None):
    # (B, C, E) -> (B, C, E)
    y = self.fc(x)
    # (B, C, E) -> (B, E, C)
    xx = einops.rearrange(x, 'b c e -> b e c')
    # (B, E, C) -> (B, E, C)
    yx = self.alt_fc(xx)
    # (B, C, E) @ (B, E, C) -> (B, C, C)
    attn = torch.einsum('bce,bek->bck', y, yx)

    if mask is not None:
      attn = attn.masked_fill(mask, float('-inf'))
    attn = self.attend(attn / math.sqrt(attn.shape[-1]))

    # (B, C, E) -> (B, C, E)
    vx = self.vfc(x)
    # (B, C, C) @ (B, C, E) -> (B, C, E)
    y = torch.einsum('bck,bke->bce', attn, vx)

    return self.post(self.post_feed(x, y))

  def extra_repr(self):
    return pyu.stri(dict(context_size=self.alt_fc.weight.shape[-1],
                         embed_size=self.fc.weight.shape[-1]))

