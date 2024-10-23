import einops.layers.torch as einpt
import py_misc_utils.alog as alog
import py_misc_utils.assert_checks as tas
import py_misc_utils.num_utils as pynu
import py_misc_utils.utils as pyu
import torch

from ... import args_sequential as aseq
from ... import conv_utils as cu
from ... import encoder_block as eb
from ... import utils as ut

from . import vit_base as vb


class ConvViT(vb.ViTBase):

  def __init__(self, shape, embed_size, num_heads, num_classes, num_layers,
               convs=None,
               attn_dropout=None,
               dropout=None,
               norm_mode=None,
               result_tiles=None,
               act=None,
               weight=None,
               label_smoothing=None):
    attn_dropout = attn_dropout or 0.1
    dropout = dropout or 0.1
    act = act or 'gelu'

    if isinstance(convs, str):
      convs = cu.convs_from_string(convs)

    cstack = cu.build_conv_stack(convs, shape=shape)
    patcher = aseq.ArgsSequential(
      cstack,
      einpt.Rearrange('b c h w -> b (h w) c'),
    )

    net = aseq.ArgsSequential(
      [eb.EncoderBlock(embed_size, num_heads,
                       attn_dropout=attn_dropout,
                       dropout=dropout,
                       norm_mode=norm_mode,
                       act=act)
       for _ in range(num_layers)])

    super().__init__(patcher, net, shape, embed_size, num_classes,
                     result_tiles=result_tiles,
                     act=act,
                     weight=weight,
                     label_smoothing=label_smoothing)

