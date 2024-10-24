import py_misc_utils.utils as pyu
import torch
import torch.nn as nn

from . import utils as ut


class StepLU(nn.Module):

  def __init__(self, threshold=None, lthreshold=None, rthreshold=None):
    super().__init__()
    if threshold is not None:
      self.lthreshold, self.rthreshold = threshold, 0.0
    else:
      self.lthreshold = pyu.value_or(lthreshold, 0.5)
      self.rthreshold = pyu.value_or(rthreshold, 0.0)

  def forward(self, x):
    zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)
    pv = torch.maximum(x - self.rthreshold, zero)
    nv = torch.minimum(x + self.lthreshold, zero)

    return torch.where(x >= zero, pv, nv)

  def extra_repr(self):
    return ut.extra_repr(lthreshold=self.lthreshold, rthreshold=self.rthreshold)

