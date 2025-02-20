import py_misc_utils.utils as pyu
import torch
import torch.nn as nn

from ... import args_sequential as aseq
from ... import layer_utils as lu
from ... import sequence_utils as sequ
from ... import shard_attention as satn

from . import sequence_base as sb


class ShardSeq(sb.SequenceBase):

  def __init__(self, context_size, embed_size, num_heads, vocab_size, num_layers,
               use_attn_mask=True,
               act='gelu',
               vocab_final='linear'):
    def post():
      return aseq.ArgsSequential(
        nn.LayerNorm(embed_size),
        lu.create(act),
      )

    super().__init__(context_size, embed_size, vocab_size)
    self.blocks = aseq.ArgsSequential(
      [satn.ShardAttention(embed_size, num_heads,
                           post=post,
                           post_feed=lambda x, y: x + y)
       for _ in range(num_layers)]
    )
    self.vocab_head = sequ.build_vocab_head(embed_size, vocab_size,
                                            act=act,
                                            final=vocab_final)
    if use_attn_mask:
      self.register_buffer('mask',
                           torch.triu(torch.ones(context_size, context_size),
                                      diagonal=1).bool(),
                           persistent=False)
    else:
      self.mask = None

  def forward(self, x, targets=None):
    y = super().forward(x)
    y = self.blocks(y, mask=self.mask)
    # (B, T, C) @ (C, V) => (B, T, V)
    y = self.vocab_head(y)

    return y, self.loss(y, targets)

