import einops
import torch
import torch.nn as nn


# The original einops layers do not support kwargs ...
class Rearrange(nn.Module):

  def __init__(self, recipe, **kwargs):
    super().__init__()
    self.recipe = recipe
    self.kwargs = kwargs

  def forward(self, x):
    return einops.rearrange(x, self.recipe, **self.kwargs)


class Repeat(nn.Module):

  def __init__(self, recipe, **kwargs):
    super().__init__()
    self.recipe = recipe
    self.kwargs = kwargs

  def forward(self, x):
    return einops.repeat(x, self.recipe, **self.kwargs)


class Reduce(nn.Module):

  def __init__(self, recipe, reduction, **kwargs):
    super().__init__()
    self.recipe = recipe
    self.reduction = reduction
    self.kwargs = kwargs

  def forward(self, x):
    return einops.reduce(x, self.recipe, self.reduction, **self.kwargs)

