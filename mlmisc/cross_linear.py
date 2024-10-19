import einops
import mlmisc.layer_utils as lu
import py_misc_utils.utils as pyu
import torch
import torch.nn as nn


class CrossLinear(nn.Module):

  def __init__(self, context_size, embed_size):
    super().__init__()
    self.fc = nn.Linear(embed_size, embed_size)
    self.alt_fc = nn.Linear(context_size, context_size)

  def forward(self, x):
    y = self.fc(x)
    xa = einops.rearrange(x, 'b c e -> b e c')
    ya = self.alt_fc(xa)
    ya = einops.rearrange(ya, 'b e c -> b c e')
    y += ya

    return y

  def extra_repr(self):
    return pyu.stri(dict(context_size=self.alt_fc.weight.shape[-1],
                         embed_size=self.fc.weight.shape[-1]))

