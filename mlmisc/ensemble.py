import torch
import torch.nn as nn

from . import args_base as ab
from . import utils as ut


class Ensemble(ab.ArgsBase):

  def __init__(self, *args, top_n_=None, **kwargs):
    super().__init__(*args, **kwargs)
    self.top_n = top_n_
    self.weight = nn.Parameter(torch.full((len(self),), 1 / len(self)))

  def forward(self, *args, **kwargs):
    if self.training and self.top_n is not None:
      rnd = torch.randn(len(self))
      weights, indices = torch.topk(rnd, self.top_n)

      y, nets = 0, list(self.values())
      for i, w in zip(indices, weights):
        y = nets[i](*args, **kwargs) * w * self.weight[i] + y

      y = y / torch.sum(weights)
    else:
      parts = [net(*args, **kwargs) * self.weight[i] for i, net in enumerate(self.values())]
      y = ut.add(*parts) / len(parts)

    return y

