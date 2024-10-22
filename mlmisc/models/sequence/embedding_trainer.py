import py_misc_utils.alog as alog
import py_misc_utils.utils as pyu
import torch
import torch.nn as nn

from ... import args_sequential as aseq
from ... import einops_layers as el
from ... import layer_utils as lu
from ... import module_builder as mb
from ... import utils as ut

from . import sequence_base as sb


def create_net_skipgram_v1(window_size, embed_size, vocab_size, act, net_kwargs):
  proj_embed_size = net_kwargs.pop('proj_embed_size', 2 * embed_size)

  net = mb.ModuleBuilder((1, embed_size))
  net.add(nn.Flatten())
  net.linear(2 * window_size * proj_embed_size)
  net.add(lu.create(act))
  net.add(el.Rearrange('b (w jes) -> b w jes', jes=proj_embed_size))
  net.linear(vocab_size)

  return net


def create_net_cbow_v1(window_size, embed_size, vocab_size, act, net_kwargs):
  proj_embed_size = net_kwargs.pop('proj_embed_size', 2 * embed_size)

  net = mb.ModuleBuilder((2 * window_size, embed_size))
  net.add(nn.Flatten())
  net.linear(proj_embed_size)
  net.add(lu.create(act))
  net.linear(vocab_size)

  return net


class EmbeddingTrainer(sb.SequenceBase):

  def __init__(self, window_size, mode, embed_size, vocab_size,
               netver=None,
               act=None,
               padding_idx=None,
               **kwargs):
    netver = pyu.value_or(netver, 'v1')
    act = pyu.value_or(act, 'relu')

    net_builder = globals()[f'create_net_{mode}_{netver}']
    net = net_builder(window_size, embed_size, vocab_size, act, kwargs)

    if kwargs:
      alog.info(f'Unused {pyu.cname(self)} keyword arguments: {kwargs}')

    context_size = 1 if mode == 'skipgram' else 2 * window_size

    super().__init__(context_size, embed_size, vocab_size,
                     use_positions=False,
                     padding_idx=padding_idx)
    self.net = net

  def forward(self, x, targets=None):
    y = super().forward(x)
    y = self.net(y)

    return y, self.loss(y, targets)

