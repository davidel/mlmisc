import collections

import py_misc_utils.assert_checks as tas
import py_misc_utils.utils as pyu
import torch
import torch.nn as nn


STATE_KEY = '_NET_STATE'

class NetBase(nn.Module):

  def device(self):
    devices = collections.defaultdict(int)
    for param in self.parameters():
      devices[param.device] += 1

    devices = sorted(devices.items(), key=lambda d: d[1])

    return devices[-1][0] if devices else torch.device('cpu')

  # The nn.Module "extra state" gets loaded after the normal state, which does not
  # allow the proper reconfiguration in case that is required before loading the
  # state parameter values.
  # The net_state_dict() and net_load_state_dict() implemented from the sub-classes
  # allows the modules to perform pre-configuration before the actual load_state_dict()
  # is called.
  def state_dict(self, *args, **kwargs):
    state = super().state_dict(*args, **kwargs)

    net_state_dict = getattr(self, 'net_state_dict', None)
    if net_state_dict is not None:
      net_state = state.get(STATE_KEY)
      if net_state is None:
        state[STATE_KEY] = net_state = dict()

      for name, value in net_state_dict().items():
        tas.check(name not in net_state, msg=f'State "{name}" already exists')
        net_state[name] = value

    return state

  def load_state_dict(self, state, *args, **kwargs):
    net_load_state_dict = getattr(self, 'net_load_state_dict', None)
    if net_load_state_dict is not None:
      net_state = state.get(STATE_KEY)

      if net_state is not None:
        net_load_state_dict(net_state)

        if not net_state:
          state.pop(STATE_KEY)

    return super().load_state_dict(state, *args, **kwargs)

  def pop_net_state(self, state, names):
    return (state.pop(name, None) for name in pyu.expand_strings(names))

