import einops.layers.torch as einpt
import py_misc_utils.alog as alog
import py_misc_utils.assert_checks as tas
import py_misc_utils.num_utils as pynu
import py_misc_utils.utils as pyu
import torch

from ... import conv_utils as cu
from ... import cross_linear as xl
from ... import layer_utils as lu
from ... import module_builder as mb
from ... import utils as ut

from . import vit_base as vb


class CrossVizSeq(vb.ViTBase):

  def __init__(self, shape, embed_size, num_classes, num_layers,
               convs=None,
               shortcut=None,
               result_tiles=None,
               act=None,
               weight=None,
               label_smoothing=None):
    shortcut = pyu.value_or(shortcut, 2)
    result_tiles = pyu.value_or(result_tiles, 2)
    act = pyu.value_or(act, 'gelu')

    patcher = vb.build_conv_patcher(convs, shape, embed_size, act)

    net = mb.ModuleBuilder((patcher.shape[0] + result_tiles, patcher.shape[1]))
    result_ids = []
    for i in range(num_layers):
      net.add(xl.CrossLinear(*net.shape),
              input_fn=mb.inputfn(result_ids, back=shortcut))
      net.add(lu.create(act))
      rid = net.layernorm()
      result_ids.append(rid)

    super().__init__(patcher, net, shape, embed_size, num_classes,
                     result_tiles=result_tiles,
                     act=act,
                     weight=weight,
                     label_smoothing=label_smoothing)

