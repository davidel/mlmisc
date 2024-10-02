import py_misc_utils.alog as alog
import py_misc_utils.utils as pyu
import torch
import torch.nn as nn

from ... import conv_utils as cu
from ... import ensemble as es
from ... import layer_utils as lu
from ... import loss_wrappers as lsw
from ... import net_base as nb
from ... import utils as ut


def _tail(num_classes, act, dropout):

  def mktail(net):
    net.add(nn.Flatten())
    net.add(nn.Dropout(dropout))
    net.linear(min(num_classes * 64, (net.shape[-1] + num_classes) // 2))
    net.add(lu.create(act))
    net.linear(num_classes)

    return net

  return mktail


def _compute_loss(loss):
  def compute(y, targets):
    return loss(y, targets)

  return compute


def _build_nets(conv_specs, num_classes, shape, act, dropout):
  tail = _tail(num_classes, act, dropout)

  nets = []
  for convs in conv_specs:
    net = cu.build_conv_stack(convs, shape=shape)
    nets.append(tail(net))

  return nets


class CNNEnsemble(nb.NetBase):

  def __init__(self, num_classes, shape,
               conv_specs=None,
               convspecs_path=None,
               num_nets=None,
               max_output=None,
               act=None,
               dropout=None,
               weight=None,
               label_smoothing=None):
    num_nets = pyu.value_or(num_nets, 8)
    max_output = pyu.value_or(max_output, 1024)
    act = pyu.value_or(act, 'relu')
    dropout = pyu.value_or(dropout, 0.2)
    label_smoothing = pyu.value_or(label_smoothing, 0.0)

    nets = []
    if not conv_specs:
      if convspecs_path is not None:
        conv_specs = cu.load_conv_specs(convspecs_path)
      else:
        tail = _tail(num_classes, act, dropout)
        conv_specs = []
        for i in range(num_nets):
          alog.debug(f'Generating CNN stack #{i}')
          net, convs = cu.create_random_stack(max_output,
                                              shape=shape,
                                              act=act,
                                              tail=tail)
          nets.append(net)
          conv_specs.append(convs)

    if not nets:
      nets = _build_nets(conv_specs, num_classes, shape, act, dropout)

    super().__init__()
    self.num_classes = num_classes
    self.shape = tuple(shape)
    self.conv_specs = tuple(conv_specs)
    self.act = act
    self.dropout = dropout
    self.loss = lsw.CatLoss(
      nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
    )
    self.nets = es.Ensemble(nets,
                            loss_fn=_compute_loss(self.loss),
                            categorical=es.VOTING)

  def forward(self, x, targets=None):
    return self.nets(x, targets=targets)

  def net_state_dict(self):
    return dict(conv_specs=self.conv_specs)

  def net_load_state_dict(self, state):
    conv_specs, = self.pop_net_state(
      state,
      ('conv_specs',)
    )

    nets = _build_nets(conv_specs, self.num_classes, self.shape, self.act,
                       self.dropout)

    self.conv_specs = conv_specs
    self.nets = es.Ensemble(nets,
                            loss_fn=_compute_loss(self.loss),
                            categorical=es.VOTING)
