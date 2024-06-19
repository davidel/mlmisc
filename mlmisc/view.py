import torch
import torch.nn as nn


class View(nn.Module):

  def __init__(self, *dims):
    super().__init__()
    self.dims = dims

  def forward(self, x):
    vdims = []
    for i in self.dims:
      if i >= 0:
        vdims.append(x.shape[i])
      else:
        vdims.append(i)

    return x.view(*vdims)

