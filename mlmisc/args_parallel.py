import torch
import torch.nn as nn

from . import args_base as ab


class ArgsParallel(ab.ArgsBase):

  def __init__(self, *args, cat_dim_=None, **kwargs):
    super().__init__(*args, **kwargs)
    self.cat_dim = cat_dim_

  def forward(self, *args, **kwargs):
    parts = [net(*args, **kwargs) for net in self.values()]

    return torch.cat(parts, dim=self.cat_dim) if self.cat_dim is not None else parts

