import py_misc_utils.assert_checks as tas
import py_misc_utils.utils as pyu
import torch
import torch.nn as nn

from ... import loss_wrappers as lsw
from ... import net_base as nb
from ... import utils as ut


class SequenceBase(nb.NetBase):

  def __init__(self, context_size, embed_size, vocab_size,
               embeddings=None,
               freeze=None,
               padding_idx=None):
    freeze = pyu.value_or(freeze, False)

    super().__init__()
    if embeddings is None:
      self.tok_emb = nn.Embedding(vocab_size, embed_size,
                                  padding_idx=padding_idx)
    else:
      tas.check_eq(tuple(embeddings.shape), (vocab_size, embed_size),
                   msg=f'Wrong embeddings shape: {tuple(embeddings.shape)} vs. ' \
                   f'{(vocab_size, embed_size)}')
      self.tok_emb = nn.Embedding.from_pretrained(embeddings,
                                                  freeze=freeze,
                                                  padding_idx=padding_idx)
    self.pos_emb = nn.Parameter(torch.zeros((1, context_size, embed_size)))
    self.loss = lsw.SeqLoss(nn.CrossEntropyLoss())

  def init(self, args):
    embedding_path = args.get('embedding_path')
    if embedding_path is not None:
      ut.torch_load_to(self.tok_emb.weight, embedding_path)

  def forward(self, x):
    y = self.tok_emb(x) + self.pos_emb

    return y

