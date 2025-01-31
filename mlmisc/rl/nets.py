import functools

import py_misc_utils.alog as alog
import torch
import torch.nn as nn

from .. import args_sequential as aseq
from .. import image_pad_concat as ipc
from .. import layer_utils as lu
from .. import module_builder as mb
from .. import net_base as nb
from .. import nn_lambda as lmbd


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


def _norm_adder(ns, x):
  return x + ns.norm


def _build_dense_layers(net, num_layers, size, act):
  for _ in range(num_layers):
    net.add(Dense(size, size, bias=False, act=act, resns=net.resns.new()))

    nsid = len(net.resns) - 2
    if nsid >= 0:
      adder = functools.partial(_norm_adder, net.resns[nsid])
      net.add(lmbd.Lambda(adder, info='Adder'))


def _build_state_net(num_states, size, act):
  net = mb.ModuleBuilder((num_states,))
  net.batchnorm1d()
  net.add(Dense(num_states, size, bias=False, act=act,
                resns=net.resns.new()))

  return net


class PiNet(nb.NetBase):

  def __init__(self, num_states, num_actions,
               min_hid_size=512,
               num_layers=3,
               act='relu'):
    hid_size = max(4 * max(num_states, num_actions), min_hid_size)

    super().__init__()
    self.net = _build_state_net(num_states, hid_size, act)

    _build_dense_layers(self.net, num_layers, hid_size, act)

    self.net.linear(num_actions, bias=False)
    self.net.layernorm()
    # NOTE: Tanh() is needed here since the actions expect a [-1, 1] output.
    self.net.add(nn.Tanh())

  def forward(self, s):
    return self.net(s)


class DRLN(nb.NetBase):

  def __init__(self, num_states, num_actions,
               min_hid_size=512,
               num_layers=4,
               act='relu'):
    hid_size = max(4 * max(num_states, num_actions), min_hid_size)
    joint_size = 2 * hid_size

    super().__init__()
    self.sentry = _build_state_net(num_states, hid_size, act)
    self.aentry = Dense(num_actions, hid_size, act=act)
    self.net = mb.ModuleBuilder((joint_size,))

    _build_dense_layers(self.net, num_layers, joint_size, act)

    self.net.linear(1)

  def forward(self, s, a):
    ya = self.aentry(a)
    ys = self.sentry(s)

    xnet = torch.cat([ys, ya], dim=-1)

    return self.net(xnet)

