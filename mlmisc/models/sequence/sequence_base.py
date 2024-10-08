import torch
import torch.nn as nn

from ... import loss_wrappers as lsw
from ... import net_base as nb


class SequenceBase(nb.NetBase):

  def __init__(self, context_size, embed_size, vocab_size,
               padding_idx=None):
    super().__init__()
    self.tok_emb = nn.Embedding(vocab_size, embed_size,
                                padding_idx=padding_idx)
    self.pos_emb = nn.Parameter(torch.zeros((1, context_size, embed_size)))
    self.loss = lsw.SeqLoss(nn.CrossEntropyLoss())

  def forward(self, x):
    y = self.tok_emb(x) + self.pos_emb

    return y

