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
from ... import loss_wrappers as lsw
from ... import utils as ut


class ViTO(nn.Module):

  def __init__(self, shape, embed_size, num_heads, num_classes, num_layers,
               htile_size=None,
               wtile_size=None,
               attn_dropout=None,
               dropout=None,
               result_tiles=None,
               act=None):
    if htile_size is None or htile_size <= 0:
      htile_size = shape[1] // pynu.nearest_divisor(shape[1], 16)
    if wtile_size is None or wtile_size <= 0:
      wtile_size = shape[2] // pynu.nearest_divisor(shape[2], 16)

    alog.info(f'Using ({htile_size}, {wtile_size}) patch sizes')

    tas.check_eq(shape[1] % htile_size, 0,
                 msg=f'H images dimension {shape[1]} must be divisible by {htile_size}')
    tas.check_eq(shape[2] % wtile_size, 0,
                 msg=f'W images dimension {shape[2]} must be divisible by {wtile_size}')

    n_htiles, n_wtiles = shape[1] // htile_size, shape[2] // wtile_size
    n_tiles = n_htiles * n_wtiles
    patch_size = shape[0] * wtile_size * htile_size

    attn_dropout = attn_dropout or 0.1
    dropout = dropout or 0.1
    result_tiles = result_tiles or 2
    act = act or 'gelu'

    super().__init__()
    self.htile_size, self.wtile_size = htile_size, wtile_size
    self.result_tiles = result_tiles
    self.loss = lsw.CatLoss(nn.CrossEntropyLoss())
    self.embedding = nn.Linear(patch_size, embed_size)
    self.pos_embedding = nn.Parameter(torch.zeros(1, n_tiles + result_tiles, embed_size))
    self.pweight = nn.Parameter(torch.zeros(result_tiles, embed_size))
    self.blocks = aseq.ArgsSequential(
      [eb.EncoderBlock(embed_size, num_heads,
                       attn_dropout=attn_dropout,
                       dropout=dropout,
                       act=act)
       for _ in range(num_layers)])

    self.prj = aseq.ArgsSequential(
      lu.create(act),
      nn.Linear(result_tiles * embed_size, embed_size, bias=False),
      nn.LayerNorm(embed_size),
      lu.create(act),
      nn.Linear(embed_size, num_classes),
    )

  def forward(self, x, targets=None):
    y = einops.rearrange(x, 'b c (nh hts) (nw wts) -> b (nh nw) (c hts wts)',
                         hts=self.htile_size,
                         wts=self.wtile_size)

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

    return y, self.loss(y, targets)

