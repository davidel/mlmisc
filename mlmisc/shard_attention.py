import math

import einops
import py_misc_utils.assert_checks as tas
import py_misc_utils.utils as pyu
import torch
import torch.nn as nn

from . import attention as atn
from . import core_utils as cu
from . import layer_utils as lu


class ShardAttention(nn.Module):

  def __init__(self, embed_size):
    super().__init__()
    self.weight = cu.kuni_parameter(embed_size, embed_size)

  def forward(self, x, mask=None):
    att = x @ torch.transpose(x, -1, -2)
    if mask is not None:
      att = att.masked_fill(mask, float('-inf'))
    att = nn.functional.softmax(att / math.sqrt(x.shape[-1]), dim=-1)

    values = x @ self.weight

    return att @ values

  def extra_repr(self):
    return cu.extra_repr(embed_size=self.weight.shape[-1])

