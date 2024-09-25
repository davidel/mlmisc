import py_misc_utils.alog as alog
import py_misc_utils.utils as pyu
import torch
import torch.nn as nn

from ... import attention as atn
from ... import einops_layers as eil
from ... import layer_utils as lu
from ... import module_builder as mb
from ... import utils as ut


def inputfn(lids, back=2):
  iid = len(lids) - back
  rid = lids[iid] if iid >= 0 else None

  def input_fn(x, results):
    return (x + results[rid]) if rid is not None else x

  return input_fn


def create_layers(shape, num_layers, embed_size, num_classes, act, dropout):
  conv_steps, min_wnd_size, attn_heads = 4, 2 * embed_size, 2

  net = mb.ModuleBuilder(shape)
  cstep = (embed_size - net.shape[0]) // conv_steps + 1
  for i in range(conv_steps):
    c, h, w = net.shape

    net.batchnorm2d()
    stride = 2 if h * w > min_wnd_size else 1
    net.conv2d(min(c + cstep, embed_size),
               kernel_size=2 * stride + 1,
               stride=stride,
               padding='valid')
    net.add(lu.create(act))

  lids = [net.last_id()]
  for i in range(num_layers):
    c, h, w = net.shape

    net.batchnorm2d(input_fn=inputfn(lids))
    net.conv2d(c, kernel_size=3, stride=1, padding='same')
    net.add(lu.create(act))
    net.add(eil.Rearrange('b c h w -> b (h w) c'))
    net.add(atn.SelfAttention(c, attn_heads, attn_dropout=dropout))
    # b (h w) (h w) -> b (h w) c
    net.linear(c)
    net.add(lu.create(act))
    net.add(nn.Dropout(dropout))
    lid = net.add(eil.Rearrange('b (h w) c -> b c h w', h=h, w=w))
    lids.append(lid)

  net.conv2d(net.shape[0], kernel_size=7, stride=2, padding='valid')
  net.add(lu.create(act))
  net.add(nn.Flatten())
  net.linear(num_classes * 32)
  net.add(lu.create(act))
  net.linear(num_classes)

  return net


class CoViT(nn.Module):

  def __init__(self, shape, num_classes, num_layers, embed_size,
               dropout=None,
               act=None,
               weight=None,
               label_smoothing=None):
    dropout = dropout if dropout is not None else 0.1
    act = act or 'relu'
    label_smoothing = label_smoothing or 0.0

    net = create_layers(shape, num_layers, embed_size, num_classes, act, dropout)

    super().__init__()
    self.loss = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
    self.net = net

  def forward(self, x, targets=None):
    y = self.net(x)

    return y, ut.compute_loss(self.loss, y, targets)

