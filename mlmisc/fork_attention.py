import math

import einops
import py_misc_utils.alog as alog
import py_misc_utils.utils as pyu
import torch
import torch.nn as nn

from . import utils as ut


def get_post_feed(post_feed):
  if isinstance(post_feed, str):
    if post_feed == 'x+y':
      return lambda x, y: x + y
    elif post_feed == 'y':
      return lambda x, y: y
    else:
      alog.xraise(ValueError, f'Invalid post-feed: {post_feed}')
  elif callable(post_feed):
    return post_feed

  return post_feed or (lambda x, y: y)


class ForkAttention(nn.Module):

  def __init__(self, context_size, embed_size,
               post_feed=None):
    super().__init__()
    self.fc = nn.Linear(embed_size, embed_size)
    self.alt_fc = nn.Linear(context_size, context_size)
    self.vfc = nn.Linear(embed_size, embed_size)
    self.attend = nn.Softmax(dim=-1)
    self.post_feed = get_post_feed(post_feed)

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

    return self.post_feed(x, y)

  def extra_repr(self):
    return ut.extra_repr(context_size=self.alt_fc.weight.shape[-1],
                         embed_size=self.fc.weight.shape[-1])

