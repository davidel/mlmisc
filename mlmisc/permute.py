import torch
import torch.nn as nn


class Permute(nn.Module):

  def __init__(self, *dims):
    super().__init__()
    self.dims = dims

  def forward(self, x):
    return torch.permute(x, self.dims)

