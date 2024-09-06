import collections

import torch
import torch.nn as nn


Pair = collections.namedtuple('Pair', 'arg, module')

# Used in APIs where, for historical reasons, there is a generic kwargs argument
# list, and where new named arguments need to be passed.
# If either an nn.Module or the new argument needs to be passed, there are passed
# as usual. In case both need to be passed, a Pair(arg=NEWARG, module=NNMODULE)
# can be used.
def extract_args(kwargs, *names):
  args, missing = [], object()
  for name in names:
    arg = kwargs.get(name)
    if isinstance(arg, nn.Module):
      arg = None
    elif isinstance(arg, Pair):
      kwargs[name] = arg.module
      arg = arg.arg
    else:
      if arg is missing:
        arg = None
      else:
        kwargs.pop(name)

    args.append(arg)

  return args[0] if len(args) == 1 else args


class ArgsParallel(nn.Module):

  def __init__(self, *args, **kwargs):
    cat_dim = extract_args(kwargs, 'cat_dim')

    super().__init__()
    self.cat_dim = cat_dim
    self.nets = nn.ModuleDict()
    net_list = args[0] if len(args) == 1 and isinstance(args[0], (list, tuple)) else args
    for i, net in enumerate(net_list):
      self.nets[f'{i}'] = net
    for name, net in kwargs.items():
      self.nets[name] = net

  def forward(self, *args, **kwargs):
    parts = [net(*args, **kwargs) for net in self.nets.values()]

    return torch.cat(parts, dim=self.cat_dim) if self.cat_dim is not None else parts

