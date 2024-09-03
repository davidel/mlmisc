import collections

import torch
import torch.nn as nn


class ArgsSequential(nn.Sequential):

  def __init__(self, *args, **kwargs):
    if not args:
      # Since from Python 3.7 keyword arguments are ordered, allow building a
      # Sequential directly from the passed keyword arguments without explicitly
      # instantiating an OrderedDict.
      super().__init__(collections.OrderedDict(kwargs))
    elif len(args) == 1 and isinstance(args[0], (list, tuple)):
      super().__init__(*args[0], **kwargs)
    else:
      super().__init__(*args, **kwargs)

  def forward(self, x, *args, **kwargs):
    for mod in self:
      x = mod(x, *args, **kwargs)

    return x

