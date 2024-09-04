import torch
import torch.nn as nn


class Lambda(nn.Module):

  def __init__(self, fn):
    super().__init__()
    self.fn = fn

  def forward(self, *args, **kwargs):
    return self.fn(*args, **kwargs)

