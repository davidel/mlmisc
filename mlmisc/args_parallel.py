import py_misc_utils.utils as pyu
import torch
import torch.nn as nn

from . import nets_dict as netd


class ArgsParallel(netd.NetsDict):

  def __init__(self, *args, **kwargs):
    cat_dim, stack_dim = pyu.pop_kwargs(kwargs, 'cat_dim, stack_dim')

    super().__init__(*args, **kwargs)
    self.cat_dim = cat_dim
    self.stack_dim = stack_dim

  def forward(self, *args, **kwargs):
    parts = [net(*args, **kwargs) for net in self.values()]

    if self.cat_dim is not None:
      return torch.cat(parts, dim=self.cat_dim)

    if self.stack_dim is not None:
      stack = [torch.unsqueeze(p, self.stack_dim) for p in parts]
      return torch.cat(stack, dim=self.stack_dim)

    return parts

