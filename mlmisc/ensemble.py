import torch
import torch.nn as nn

from . import args_parallel as ap
from . import utils as ut


class Ensemble(ap.ArgsParallel):

  def __init__(self, *args, top_n=None, **kwargs):
    super().__init__(*args, **kwargs)
    self.top_n = top_n

  def forward(self, *args, **kwargs):
    if self.training and self.top_n is not None:
      rnd = torch.randn(len(self.nets))
      weights, indices = torch.topk(rnd, self.top_n)

      y, nets = 0, list(self.nets.values())
      for i, w in zip(indices, weights):
        y = nets[i](*args, **kwargs) * w + y

      y = y / torch.sum(weights)
    else:
      parts = [net(*args, **kwargs) for net in self.nets.values()]
      y = ut.add(*parts) / len(parts)

    return y

