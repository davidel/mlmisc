import einops.layers.torch as einpt
import py_misc_utils.alog as alog
import py_misc_utils.assert_checks as tas
import py_misc_utils.num_utils as pynu
import py_misc_utils.utils as pyu
import torch

from ... import args_sequential as aseq
from ... import conv_utils as cu
from ... import utils as ut

from . import vit_base as vb


class ConvViT(vb.ViTBase):

  def __init__(self, shape, embed_size, num_heads, num_classes, num_layers,
               convs=None,
               attn_dropout=None,
               dropout=None,
               norm_mode=None,
               patch_mode=None,
               result_tiles=None,
               act=None,
               weight=None,
               label_smoothing=None):
    if isinstance(convs, str):
      convs = cu.convs_from_string(convs)

    cstack = cu.build_conv_stack(convs, shape=shape)
    patcher = aseq.ArgsSequential(
      cstack,
      einpt.Rearrange('b c h w -> b (h w) c'),
    )

    super().__init__(ut.net_shape(patcher, shape), embed_size, num_heads, num_classes, num_layers,
                     patch_specs=patch_specs,
                     attn_dropout=attn_dropout,
                     dropout=dropout,
                     norm_mode=norm_mode,
                     patch_mode=patch_mode,
                     result_tiles=result_tiles,
                     act=act,
                     weight=weight,
                     label_smoothing=label_smoothing)
    self.patcher = patcher

