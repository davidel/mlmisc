import py_misc_utils.alog as alog
import torch
import torch.nn as nn

from .. import args_sequential as aseq
from .. import layer_utils as lu
from .. import module_builder as mb
from .. import net_base as nb
from .. import nn_lambda as lmbd
from .. import results_namespace as rns


class Dense(nb.NetBase):

  def __init__(self, in_features, out_features,
               bias=True,
               act='relu',
               resns=None):
    super().__init__(result_ns=resns)
    self.fc = nn.Linear(in_features, out_features, bias=bias)
    self.norm = nn.LayerNorm(out_features)
    self.act = lu.create(act)

  def forward(self, x):
    y_fc = self.fc(x)
    y_norm = self.norm(y_fc)
    y_act = self.act(y_norm)

    self.set_result(fc=y_fc, norm=y_norm, act=y_act)

    return y_act


def _build_dense_layers(net, rns, num_layers, size, act):
  for _ in range(num_layers):
    net.add(Dense(size, size, bias=False, act=act, resns=rns.ns_new()))

    nsid = rns.ns_len() - 2
    if nsid >= 0:
      net.add(lmbd.Lambda(lambda x, rns=rns.ns_get(nsid): x + rns.norm, info='Adder'))


def _compute_hidden_size(shape, num_actions, min_hid_size):
  if len(shape) == 1:
    state_size = shape[0]
  else:
    pass

  return max(4 * max(state_size, num_actions), min_hid_size)


def _build_state_net(shape, size, act, rns=None):
  if len(shape) == 1:
    net = mb.ModuleBuilder(shape)
    net.batchnorm1d()
    net.add(Dense(shape[0], size, bias=False, act=act,
                  resns=rns.ns_new() if rns is not None else None))
  else:
    pass

  return net


class PiNet(nb.NetBase):

  def __init__(self, state_shape, num_actions,
               min_hid_size=512,
               num_layers=3,
               act='relu'):
    hid_size = _compute_hidden_size(state_shape, num_actions, min_hid_size)

    super().__init__()
    self.rns = rns.ResultsNamespace()

    net = _build_state_net(state_shape, hid_size, act, rns=self.rns)
    _build_dense_layers(net, self.rns, num_layers, hid_size, act)
    net.linear(num_actions, bias=False)
    net.layernorm()
    # NOTE: Tanh() is needed here since the actions expect a [-1, 1] output.
    net.add(nn.Tanh())

    self.rns.add_net(net, name='pinet')

  def forward(self, s):
    return self.rns(s)


class DRLN(nb.NetBase):

  def __init__(self, state_shape, num_actions,
               min_hid_size=512,
               num_layers=4,
               act='relu'):
    hid_size = _compute_hidden_size(state_shape, num_actions, min_hid_size)
    joint_size = 2 * hid_size

    super().__init__()
    self.sentry = _build_state_net(state_shape, hid_size, act)
    self.aentry = Dense(num_actions, hid_size, act=act)
    self.rns = rns.ResultsNamespace()

    net = mb.ModuleBuilder((joint_size,))
    _build_dense_layers(net, self.rns, num_layers, joint_size, act)
    net.linear(1)

    self.rns.add_net(net, name='qnet')

  def forward(self, s, a):
    ya = self.aentry(a)
    ys = self.sentry(s)

    xnet = torch.cat([ys, ya], dim=-1)

    return self.rns(xnet)

