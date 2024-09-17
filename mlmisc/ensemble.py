import torch
import torch.nn as nn

from . import utils as ut


class Ensemble(nn.Module):

  def __init__(self, nets, loss_fn=None, categorical=None):
    super().__init__()
    self.nets = nn.ModuleList(nets)
    self.loss_fn = loss_fn
    self.categorical = categorical

  def forward(self, *args, targets=None, **kwargs):
    parts = [net(*args, **kwargs) for net in self.nets]

    if self.categorical == 'voting':
      y = torch.zeros_like(parts[0])
      for p in parts:
        tops = torch.argmax(p, dim=-1)
        y += torch.nn.functional.one_hot(tops, num_classes=p.shape[-1]).to(y.dtype)
    else:
      y = ut.add(parts) / len(parts)

    if self.loss_fn is None:
      return y

    loss = None
    if targets is not None:
      if self.training:
        loss = ut.add(self.loss_fn(p, targets) for p in parts) / len(parts)
      else:
        loss = self.loss_fn(y, targets)

    return y, loss

