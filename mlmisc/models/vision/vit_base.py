import einops
import py_misc_utils.alog as alog
import py_misc_utils.assert_checks as tas
import py_misc_utils.num_utils as pynu
import py_misc_utils.utils as pyu
import torch
import torch.nn as nn

from ... import args_sequential as aseq
from ... import encoder_block as eb
from ... import layer_utils as lu
from ... import patcher as pch
from ... import utils as ut


class ViTBase(nn.Module):

  def __init__(self, shape, embed_size, num_heads, num_classes, num_layers,
               attn_dropout=None,
               dropout=None,
               norm_mode=None,
               patch_mode=None,
               result_tiles=None,
               act=None,
               weight=None,
               label_smoothing=None):
    attn_dropout = attn_dropout or 0.1
    dropout = dropout or 0.1
    result_tiles = result_tiles or 2
    act = act or 'gelu'
    label_smoothing = label_smoothing or 0.0

    n_tiles, patch_size = shape

    super().__init__()
    self.result_tiles = result_tiles
    self.loss = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
    self.embedding = nn.Linear(patch_size, embed_size)
    self.pos_embedding = nn.Parameter(torch.zeros(1, n_tiles + result_tiles, embed_size))
    self.pweight = nn.Parameter(torch.zeros(result_tiles, embed_size))
    self.blocks = aseq.ArgsSequential(
      [eb.EncoderBlock(embed_size, num_heads,
                       attn_dropout=attn_dropout,
                       dropout=dropout,
                       norm_mode=norm_mode,
                       act=act)
       for _ in range(num_layers)])

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
    y = self.blocks(y)
    # (B, NH * NW + RT, E) => (B, RT, E)
    y = y[:, : self.result_tiles, :]
    y = einops.rearrange(y, 'b rt e -> b (rt e)')
    # (B, RT * E) => (B, NC)
    y = self.prj(y)

    return y, ut.compute_loss(self.loss, y, targets)

