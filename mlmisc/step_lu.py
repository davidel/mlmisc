import torch
import torch.nn as nn


class StepLU(nn.Module):

  def __init__(self, threshold=None, lthreshold=None, rthreshold=None):
    super().__init__()
    if threshold is not None:
      self.lthreshold, self.rthreshold = threshold, threshold
    else:
      self.lthreshold = lthreshold or 0.25
      self.rthreshold = rthreshold or 0.25

  def forward(self, x):
    zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)
    pv = torch.maximum(x - self.rthreshold, zero)
    nv = torch.minimum(x + self.lthreshold, zero)

    return torch.where(x >= zero, pv, nv)

