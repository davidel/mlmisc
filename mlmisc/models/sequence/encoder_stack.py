import py_misc_utils.utils as pyu
import torch
import torch.nn as nn

from ... import attention as atn
from ... import config as conf
from ... import encoder_block as eb
from ... import layer_utils as lu
from ... import module_builder as mb
from ... import sequence_utils as sequ

from . import sequence_base as sb


class EncoderStack(sb.SequenceBase):

  def __init__(self, context_size, embed_size, num_heads, vocab_size, num_layers,
               use_attn_mask=True,
               attn_dropout=0.0,
               dropout=0.0,
               norm_mode=eb.PRE_NORM,
               act='gelu',
               vocab_final='linear'):
    super().__init__(context_size, embed_size, vocab_size)

    net = mb.ModuleBuilder((context_size, embed_size))
    for _ in range(num_layers):
      net.add(eb.EncoderBlock(embed_size, num_heads,
                              attn_dropout=attn_dropout,
                              dropout=dropout,
                              act=act,
                              norm_mode=norm_mode),
              net_args=('mask',))

    net.add(sequ.build_vocab_head(embed_size, vocab_size,
                                  act=act,
                                  final=vocab_final))

    self.net = net
    if use_attn_mask:
      self.register_buffer('mask', sequ.causal_mask(context_size), persistent=False)
    else:
      self.mask = None

  def forward(self, x, targets=None):
    y = super().forward(x)
    y = self.net(y, mask=atn.clip_mask(y, self.mask))

    return y, self.loss(y, targets)

