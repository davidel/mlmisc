import py_misc_utils.alog as alog
import py_misc_utils.assert_checks as tas
import py_misc_utils.inspect_utils as pyiu
import py_misc_utils.num_utils as pynu
import py_misc_utils.utils as pyu
import torch
import torch.nn as nn

from ... import cross_linear as xl
from ... import layer_utils as lu
from ... import module_builder as mb
from ... import sequence_utils as sequ

from . import sequence_base as sb


def create_net(context_size, embed_size, vocab_size, num_layers, shortcut, act):
  net = mb.ModuleBuilder((context_size, embed_size))
  result_ids = []
  for i in range(num_layers):
    rid = net.add(xl.CrossLinear(context_size, embed_size),
                  net_args=('mask',),
                  input_fn=mb.inputsum_back(result_ids, back=shortcut))
    result_ids.append(rid)
    net.layernorm()
    net.add(lu.create(act))

  return net


class CrossSeq(sb.SequenceBase):

  def __init__(self, context_size, embed_size, vocab_size,
               use_attn_mask=True,
               num_layers=8,
               shortcut=2,
               act='relu',
               padding_idx=None,
               vocab_final='linear'):
    net = create_net(context_size, embed_size, vocab_size, num_layers, shortcut, act)

    super().__init__(context_size, embed_size, vocab_size,
                     padding_idx=padding_idx)
    self.net = net
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
    y = self.net(y, mash=self.mask)
    # (B, T, C) @ (C, V) => (B, T, V)
    y = self.vocab_head(y)

    return y, self.loss(y, targets)

