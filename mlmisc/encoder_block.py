import py_misc_utils.assert_checks as tas
import py_misc_utils.utils as pyu
import torch
import torch.nn as nn

from . import args_sequential as aseq
from . import attention as atn
from . import layer_utils as lu
from . import utils as ut


PRE_NORM = 'pre'
POST_NORM = 'post'
NORM_MODES = {PRE_NORM, POST_NORM}


class EncoderBlock(nn.Module):

  def __init__(self, input_dim, num_heads,
               dim_feedforward=None,
               attn_dropout=None,
               dropout=None,
               act=None,
               norm_mode=None):
    dim_feedforward = pyu.value_or(dim_feedforward, 4 * input_dim)
    attn_dropout = pyu.value_or(attn_dropout, 0.0)
    dropout = pyu.value_or(dropout, 0.0)
    act = pyu.value_or(act, nn.GELU)
    norm_mode = pyu.value_or(norm_mode, PRE_NORM)

    tas.check(norm_mode in NORM_MODES,
              msg=f'Unknown norm mode "{norm_mode}" (should be one of {NORM_MODES})')

    super().__init__()
    self.norm_mode = norm_mode
    self.attn = atn.SelfAttention(input_dim, num_heads,
                                  dropout=attn_dropout)

    self.linear_net = aseq.ArgsSequential(
      ifc=nn.Linear(input_dim, dim_feedforward),
      act=lu.create(act),
      ofc=nn.Linear(dim_feedforward, input_dim),
      odrop=nn.Dropout(dropout),
    )

    self.norm1 = nn.LayerNorm(input_dim)
    self.norm2 = nn.LayerNorm(input_dim)

  def forward(self, x, mask=None):
    if self.norm_mode == PRE_NORM:
      xx = self.norm1(x)
      x = x + self.attn(xx, mask=mask)
      x = x + self.linear_net(self.norm2(x))
    else:
      x = self.norm1(x + self.attn(x, mask=mask))
      x = self.norm2(x + self.linear_net(x))

    return x

  def extra_repr(self):
    return ut.extra_repr(norm_mode=self.norm_mode)

