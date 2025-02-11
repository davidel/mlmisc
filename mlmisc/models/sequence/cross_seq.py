import py_misc_utils.alog as alog
import py_misc_utils.assert_checks as tas
import py_misc_utils.inspect_utils as pyiu
import py_misc_utils.num_utils as pynu
import py_misc_utils.utils as pyu
import torch
import torch.nn as nn

from ... import cross_linear as xl
from ... import einops_layers as eil
from ... import layer_utils as lu
from ... import module_builder as mb
from ... import tiled_linear as tl

from . import sequence_base as sb


def create_net_v1(context_size, embed_size, vocab_size, num_layers, bottleneck,
                  act, dropout, net_kwargs):
  shortcut = net_kwargs.pop('shortcut', 3)

  net = mb.ModuleBuilder((context_size, embed_size))
  result_ids = []
  for i in range(num_layers):
    net.add(xl.CrossLinear(context_size, embed_size),
            input_fn=mb.inputsum_back(result_ids, back=shortcut))
    net.add(lu.create(act))
    rid = net.layernorm()
    result_ids.append(rid)

  net.add(nn.Flatten())
  net.add(nn.Dropout(dropout))
  net.linear(bottleneck)
  net.add(lu.create(act))
  net.linear(vocab_size)

  return net


def create_net_v2(context_size, embed_size, vocab_size, num_layers, bottleneck,
                  act, dropout, net_kwargs):
  shortcut = net_kwargs.pop('shortcut', 3)
  num_tiles = net_kwargs.pop('num_tiles', 16)
  crossed = net_kwargs.pop('crossed', False)

  net = mb.ModuleBuilder((context_size, embed_size))
  result_ids = []
  for i in range(num_layers):
    net.add(xl.CrossLinear(context_size, embed_size),
            input_fn=mb.inputsum_back(result_ids, back=shortcut))
    net.add(lu.create(act))
    rid = net.layernorm()
    result_ids.append(rid)

  net.add(eil.Rearrange('b c e -> b (e c)'))
  net.add(nn.Dropout(dropout))
  net.add(tl.TiledLinear(net.shape[-1], bottleneck, num_tiles, crossed=crossed))
  net.add(lu.create(act))
  net.linear(vocab_size)

  return net


class CrossSeq(sb.SequenceBase):

  def __init__(self, context_size, embed_size, vocab_size,
               netver='v1',
               num_layers=8,
               bottleneck=None,
               act='relu',
               dropout=0.1,
               padding_idx=None,
               **kwargs):
    bottleneck = pyu.value_or(bottleneck, pynu.round_up(vocab_size // 8, 128))

    net_builder = globals()[f'create_net_{netver}']
    net = net_builder(context_size,
                      embed_size,
                      vocab_size,
                      num_layers,
                      bottleneck,
                      act,
                      dropout,
                      kwargs)
    if kwargs:
      alog.info(f'Unused {pyiu.cname(self)} keyword arguments: {kwargs}')

    super().__init__(context_size, embed_size, vocab_size,
                     padding_idx=padding_idx)
    self.net = net

  def forward(self, x, targets=None):
    y = super().forward(x)
    y = self.net(y)

    return y, self.loss(y, targets)

