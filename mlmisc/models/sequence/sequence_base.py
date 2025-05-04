import py_misc_utils.assert_checks as tas
import py_misc_utils.utils as pyu
import torch
import torch.nn as nn

from ... import core_utils as cu
from ... import loss_wrappers as lsw
from ... import net_base as nb


class SequenceBase(nb.NetBase):

  def __init__(self, context_size, embed_size, vocab_size,
               use_positions=True,
               padding_idx=None):
    super().__init__()
    self.tok_emb = nn.Embedding(vocab_size, embed_size,
                                padding_idx=padding_idx)
    self.pos_emb = nn.Parameter(torch.zeros((1, context_size, embed_size))) \
      if use_positions else None
    self.loss = lsw.SeqLoss(nn.CrossEntropyLoss())

  def forward(self, x):
    y = self.tok_emb(x)
    if self.pos_emb is not None:
      context_size = y.shape[-2]
      y = y + self.pos_emb[..., : context_size, :]

    return y

