import einops.layers.torch as einpt
import py_misc_utils.alog as alog
import py_misc_utils.assert_checks as tas
import py_misc_utils.num_utils as pynu
import py_misc_utils.utils as pyu
import torch

from ... import fork_attention as fa
from ... import layer_utils as lu
from ... import module_builder as mb
from ... import utils as ut

from . import vit_base as vb


class ForkVizSeq(vb.ViTBase):

  def __init__(self, shape, embed_size, num_classes, num_layers, convs,
               result_tiles=2,
               act='gelu',
               weight=None,
               label_smoothing=None):
    patcher = vb.build_conv_patcher(convs, shape, embed_size, act)

    net = mb.ModuleBuilder((patcher.shape[0] + result_tiles, patcher.shape[1]))
    for _ in range(num_layers):
      net.add(fa.ForkAttention(*net.shape, post_feed='x+y'))
      net.add(lu.create(act))
      net.layernorm()

    super().__init__(patcher, net, shape, embed_size, num_classes,
                     result_tiles=result_tiles,
                     act=act,
                     weight=weight,
                     label_smoothing=label_smoothing)

