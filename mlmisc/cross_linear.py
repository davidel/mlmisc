import einops
import py_misc_utils.utils as pyu
import torch
import torch.nn as nn

from . import utils as ut


class OrigCrossLinear(nn.Module):

  def __init__(self, context_size, embed_size, bias=None):
    bias = pyu.value_or(bias, True)

    super().__init__()
    self.fc = nn.Parameter(ut.kuni_tensor((embed_size, embed_size)))
    self.alt_fc = nn.Parameter(ut.kuni_tensor((context_size, context_size)))
    self.bias = nn.Parameter(torch.zeros(embed_size)) if bias else None

  def forward(self, x):
    y = x @ self.fc
    y = y + torch.einsum('bce,ck->bke', x, self.alt_fc)
    if self.bias is not None:
      y = y + self.bias

    return y

  def extra_repr(self):
    return ut.extra_repr(context_size=self.alt_fc.shape[-1],
                         embed_size=self.fc.shape[-1],
                         bias=self.bias is not None)


class CrossLinear(nn.Module):

  def __init__(self, context_size, embed_size, bias=None):
    bias = pyu.value_or(bias, True)

    super().__init__()
    self.fc = nn.Parameter(ut.kuni_tensor((embed_size, embed_size)))
    self.alt_fc = nn.Parameter(ut.kuni_tensor((context_size, embed_size)))
    self.bias = nn.Parameter(torch.zeros(embed_size)) if bias else None

  def forward(self, x):
    y = x @ self.fc
    ya = torch.einsum('bce,ck->bke', x, self.alt_fc)
    y = torch.einsum('bce,bek->bck', y, ya)
    if self.bias is not None:
      y = y + self.bias

    return y

  def extra_repr(self):
    return ut.extra_repr(context_size=self.alt_fc.shape[-1],
                         embed_size=self.fc.shape[-1],
                         bias=self.bias is not None)

