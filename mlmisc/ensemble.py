import torch
import torch.nn as nn

from . import core_utils as cu


VOTING = 'voting'

class Ensemble(nn.Module):

  def __init__(self, nets, loss_fn=None, categorical=None):
    super().__init__()
    self.nets = nn.ModuleList(nets)
    self.loss_fn = loss_fn
    self.categorical = categorical

  def forward(self, *args, targets=None, **kwargs):
    parts = [net(*args, **kwargs) for net in self.nets]

    if self.categorical == VOTING:
      y = torch.zeros_like(parts[0])
      for p in parts:
        tops = torch.argmax(p, dim=-1)
        y += torch.nn.functional.one_hot(tops, num_classes=p.shape[-1]).to(y.dtype)
    else:
      y = cu.add(*parts) / len(parts)

    if self.loss_fn is None:
      return y

    loss = None
    if targets is not None:
      if self.training:
        loss = cu.add(*[self.loss_fn(p, targets) for p in parts]) / len(parts)
      else:
        loss = self.loss_fn(y, targets)

    return y, loss

