import py_misc_utils.utils as pyu
import torch
import torch.nn as nn

from . import utils as ut


class StepLU(nn.Module):

  def __init__(self, threshold=None):
    super().__init__()
    if isinstance(threshold, (list, tuple)):
      self.lthreshold, self.rthreshold = threshold
    else:
      self.lthreshold, self.rthreshold = pyu.value_or(threshold, 0.5), 0.0

  def forward(self, x):
    zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)
    pv = torch.maximum(x - self.rthreshold, zero)
    nv = torch.minimum(x + self.lthreshold, zero)

    return torch.where(x >= zero, pv, nv)

  def extra_repr(self):
    return ut.extra_repr(lthreshold=self.lthreshold, rthreshold=self.rthreshold)

