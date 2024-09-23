import py_misc_utils.assert_checks as tas
import py_misc_utils.utils as pyu
import torch
import torch.nn as nn


STATE_KEY = '_NET_STATE'

class NetBase(nn.Module):

  def __init__(self, device=None):
    super().__init__()
    self.device = torch.device(device or 'cpu')

  def to(self, *args, **kwargs):
    if args and isinstance(args[0], (str, torch.device)):
      device = torch.device(args[0])
    else if 'device' in kwargs:
      device = torch.device(kwargs['device'])
    else:
      device = self.device

    result = super().to(*args, **kwargs)

    self.device = device

    return result

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

      net_load_state_dict(net_state)

      if not net_state:
        state.pop(STATE_KEY)

    return super().load_state_dict(state, *args, **kwargs)

  def pop_net_state(self, state, names):
    return (state.pop(name, None) for name in pyu.as_sequence(names))

