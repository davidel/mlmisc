import torch
import torch.nn as nn


class StepLU(nn.Module):

  def __init__(self, threshold=None):
    super().__init__()
    self.threshold = threshold or 0.5

  def forward(self, x):
    zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)
    pv = torch.maximum(x - self.threshold, zero)
    nv = torch.minimum(x + self.threshold, zero)

    return torch.where(x >= zero, pv, nv)

