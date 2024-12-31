import math

import numpy as np
import py_misc_utils.alog as alog
import py_misc_utils.num_utils as pynu
import py_misc_utils.utils as pyu
import torch
import torch.nn as nn

from ... import attention as atn
from ... import einops_layers as eil
from ... import layer_utils as lu
from ... import loss_wrappers as lsw
from ... import module_builder as mb
from ... import net_base as nb
from ... import utils as ut


def create_layers(shape, num_layers, embed_size, num_patches, num_classes,
                  act, dropout):
  # PixelsPerPatch = np.prod(shape[1:]) / num_patches
  # log2(PixelsPerPatch) / 2 == log4(PixelsPerPatch) ... (every step reduces by 2x2).
  conv_steps = max(1, int(math.log2(np.prod(shape[1:]) / num_patches) / 2))
  patches_x_edge = round(math.sqrt(num_patches))
  attn_heads = 2
  num_classes_amp = 16

  net = mb.ModuleBuilder(shape)
  cstep = (embed_size - net.shape[0]) // conv_steps + 1
  for i in range(conv_steps):
    c, h, w = net.shape

    hstride = 2 if (h - 5) >= 2 * patches_x_edge else 1
    wstride = 2 if (w - 5) >= 2 * patches_x_edge else 1
    hkernel_size = min(h, 2 * hstride + 1)
    wkernel_size = min(w, 2 * wstride + 1)
    channels = min(pynu.round_up(c + cstep, 8), embed_size)

    net.batchnorm2d()
    net.conv2d(channels, (hkernel_size, wkernel_size),
               stride=(hstride, wstride),
               padding='valid')
    net.add(lu.create(act))

  lids = [net.last_id()]
  for i in range(num_layers):
    c, h, w = net.shape

    net.batchnorm2d(input_fn=mb.inputfn(lids))
    net.conv2d(c, 3, stride=1, padding='same')
    net.add(lu.create(act))
    net.add(eil.Rearrange('b c h w -> b (h w) c'))
    # b (h w) c -> b (h w) (h w)
    net.add(atn.SelfAttention(c, attn_heads, attn_dropout=dropout))
    # b (h w) (h w) -> b (h w) c
    net.linear(c)
    net.add(lu.create(act))
    net.add(nn.Dropout(dropout))
    lid = net.add(eil.Rearrange('b (h w) c -> b c h w', h=h, w=w))
    lids.append(lid)

  net.conv2d(net.shape[0], 5, stride=2, padding='valid')
  net.add(lu.create(act))
  net.add(nn.Flatten())
  net.linear(num_classes * num_classes_amp)
  net.add(lu.create(act))
  net.linear(num_classes)

  return net


class CoViT(nb.NetBase):

  def __init__(self, shape, num_classes, num_layers, embed_size, num_patches,
               dropout=None,
               act=None,
               weight=None,
               label_smoothing=None):
    dropout = pyu.value_or(dropout, 0.1)
    act = pyu.value_or(act, nn.ReLU)
    label_smoothing = pyu.value_or(label_smoothing, 0.0)

    net = create_layers(shape, num_layers, embed_size, num_patches, num_classes,
                        act, dropout)

    super().__init__()
    self.loss = lsw.CatLoss(
      nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
    )
    self.net = net

  def forward(self, x, targets=None):
    y = self.net(x)

    return y, self.loss(y, targets)

