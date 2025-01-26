import py_misc_utils.alog as alog
import torch
import torch.nn as nn

from .. import args_sequential as aseq
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


def _compute_hidden_size(shape, num_actions, min_hid_size):
  if len(shape) == 1:
    state_size = shape[0]
  elif len(shape) == 3:
    pass
  else:
    alog.xraise(ValueError, f'Invalid state network shape: {shape}')

  return max(4 * max(state_size, num_actions), min_hid_size)


def _build_dense_layers(net, num_layers, size, act):
  resns = net.resns
  for _ in range(num_layers):
    net.add(Dense(size, size, bias=False, act=act, resns=resns.new()))

    nsid = len(resns) - 2
    if nsid >= 0:
      net.add(lmbd.Lambda(lambda x, ns=resns[nsid]: x + ns.norm, info='Adder'))


def _build_state_net(shape, size, act):
  if len(shape) == 1:
    net = mb.ModuleBuilder(shape)
    net.batchnorm1d()
    net.add(Dense(shape[0], size, bias=False, act=act,
                  resns=net.resns.new()))
  elif len(shape) == 3:
    pass
  else:
    alog.xraise(ValueError, f'Invalid state network shape: {shape}')

  return net


class PiNet(nb.NetBase):

  def __init__(self, state_shape, num_actions,
               min_hid_size=512,
               num_layers=3,
               act='relu'):
    hid_size = _compute_hidden_size(state_shape, num_actions, min_hid_size)

    super().__init__()
    self.net = _build_state_net(state_shape, hid_size, act)

    _build_dense_layers(self.net, num_layers, hid_size, act)

    self.net.linear(num_actions, bias=False)
    self.net.layernorm()
    # NOTE: Tanh() is needed here since the actions expect a [-1, 1] output.
    self.net.add(nn.Tanh())

  def forward(self, s):
    return self.net(s)


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
    self.net = mb.ModuleBuilder((joint_size,))

    _build_dense_layers(self.net, num_layers, joint_size, act)

    self.net.linear(1)

  def forward(self, s, a):
    ya = self.aentry(a)
    ys = self.sentry(s)

    xnet = torch.cat([ys, ya], dim=-1)

    return self.net(xnet)

