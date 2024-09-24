import py_misc_utils.alog as alog
import py_misc_utils.assert_checks as tas
import py_misc_utils.num_utils as pynu
import py_misc_utils.utils as pyu
import torch

from ... import patcher as pch
from ... import utils as ut

from . import vit_base as vb


class PatcherViT(vb.ViTBase):

  def __init__(self, shape, embed_size, num_heads, num_classes, num_layers,
               patch_specs=None,
               attn_dropout=None,
               dropout=None,
               norm_mode=None,
               patch_mode=None,
               result_tiles=None,
               act=None,
               weight=None,
               label_smoothing=None):
    patcher_config = []
    if patch_specs:
      for pcfg in patch_specs.split(':'):
        patch_args = pyu.parse_dict(pcfg)
        patcher_config.append(pch.Patch(**patch_args))
    else:
      hsize = shape[1] // pynu.nearest_divisor(shape[1], 16)
      wsize = shape[2] // pynu.nearest_divisor(shape[2], 16)
      alog.info(f'Using ({hsize}, {wsize}) patch sizes')

      patcher_config.append(pch.Patch(hsize=hsize, wsize=wsize, hstride=hsize, wstride=wsize))
      patcher_config.append(pch.Patch(hsize=hsize, wsize=wsize, hstride=hsize, wstride=wsize,
                                      hbase=hsize // 2, wbase=wsize // 2))

    patcher = pch.Patcher(patcher_config,
                          mode=patch_mode,
                          in_channels=shape[0])
    n_tiles, patch_size = ut.net_shape(patcher, shape)

    super().__init__(ut.net_shape(patcher, shape), embed_size, num_heads, num_classes, num_layers,
                     attn_dropout=attn_dropout,
                     dropout=dropout,
                     norm_mode=norm_mode,
                     patch_mode=patch_mode,
                     result_tiles=result_tiles,
                     act=act,
                     weight=weight,
                     label_smoothing=label_smoothing)
    self.patcher = patcher

