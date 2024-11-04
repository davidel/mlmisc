import einops
import py_misc_utils.alog as alog
import py_misc_utils.assert_checks as tas
import py_misc_utils.num_utils as pynu
import py_misc_utils.utils as pyu
import torch
import torch.nn as nn

from ... import args_sequential as aseq
from ... import conv_utils as cu
from ... import layer_utils as lu
from ... import loss_wrappers as lsw
from ... import net_base as nb
from ... import utils as ut


def build_conv_patcher(convs, shape, embed_size, act):
  if isinstance(convs, str):
    convs = cu.convs_from_string(convs)

  patcher = cu.build_conv_stack(convs, shape=shape)
  patcher.add(einpt.Rearrange('b c h w -> b (h w) c'))
  patcher.linear(embed_size)
  patcher.add(lu.create(act))

  alog.debug(f'Input shape {shape}, patcher shape {patcher.shape}')

  return patcher


class ViTBase(nb.NetBase):

  def __init__(self, patcher, net, ishape, embed_size, num_classes,
               result_tiles=None,
               act=None,
               weight=None,
               label_smoothing=None):
    result_tiles = pyu.value_or(result_tiles, 2)
    act = pyu.value_or(act, 'gelu')
    label_smoothing = pyu.value_or(label_smoothing, 0.0)

    shape = ut.net_shape(patcher, ishape)

    n_tiles, patch_size = shape

    alog.debug(f'ViT using {n_tiles} tiles of size {patch_size}')

    super().__init__()
    self.patcher = patcher
    self.net = net
    self.result_tiles = result_tiles
    self.loss = lsw.CatLoss(
      nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
    )
    self.embedding = nn.Linear(patch_size, embed_size)
    self.pos_embedding = nn.Parameter(torch.zeros(1, n_tiles + result_tiles, embed_size))
    self.pweight = nn.Parameter(torch.zeros(result_tiles, embed_size))

    self.prj = aseq.ArgsSequential(
      nn.LayerNorm(embed_size * result_tiles),
      nn.Linear(embed_size * result_tiles, embed_size, bias=False),
      lu.create(act),
      nn.Linear(embed_size, num_classes),
    )

  def forward(self, x, targets=None):
    y = self.patcher(x)
    # (B, NH * NW, C * HS * WS) => (B, NH * NW, E)
    y = self.embedding(y)
    pw = einops.repeat(self.pweight, 'rt e -> b rt e', b=x.shape[0])
    # (B, RT, E) + (B, NH * NW, E) => (B, NH * NW + RT, E)
    y = torch.cat((pw, y), dim=1)
    y = y + self.pos_embedding
    # (B, NH * NW + RT, E) => (B, NH * NW + RT, E)
    y = self.net(y)
    # (B, NH * NW + RT, E) => (B, RT, E)
    y = y[:, : self.result_tiles, :]
    y = einops.rearrange(y, 'b rt e -> b (rt e)')
    # (B, RT * E) => (B, NC)
    y = self.prj(y)

    return y, self.loss(y, targets)

