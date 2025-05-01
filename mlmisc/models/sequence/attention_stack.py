import py_misc_utils.utils as pyu
import torch
import torch.nn as nn

from ... import args_sequential as aseq
from ... import config as conf
from ... import layer_utils as lu
from ... import module_builder as mb
from ... import sequence_utils as sequ
from ... import shard_attention as satn

from . import sequence_base as sb


def create_net(attn_config, context_size, embed_size, num_heads, num_layers,
               shortcut, act):
  net = mb.ModuleBuilder((context_size, embed_size))
  result_ids = [net.last_id()]
  for i in range(num_layers):
    attn = conf.create_object('Attention Layer', attn_config, embed_size, num_heads)
    rid = net.add(attn,
                  net_args=('mask',),
                  input_fn=mb.inputsum_back(result_ids, back=shortcut))
    result_ids.append(rid)
    net.layernorm()
    net.add(lu.create(act))

  return net


class AttentionStack(sb.SequenceBase):

  def __init__(self, context_size, embed_size, num_heads, vocab_size, num_layers,
               attn_config='mlmisc.attention.create:is_self=True',
               use_attn_mask=True,
               shortcut=1,
               act='gelu',
               vocab_final='linear'):
    super().__init__(context_size, embed_size, vocab_size)

    self.net = create_net(attn_config, context_size, embed_size, num_heads, num_layers,
                          shortcut, act)
    self.net.add(sequ.build_vocab_head(embed_size, vocab_size,
                                       act=act,
                                       final=vocab_final))
    if use_attn_mask:
      self.register_buffer('mask', sequ.causal_mask(context_size), persistent=False)
    else:
      self.mask = None

  def forward(self, x, targets=None):
    y = super().forward(x)
    y = self.net(y, mask=self.mask)

    return y, self.loss(y, targets)

