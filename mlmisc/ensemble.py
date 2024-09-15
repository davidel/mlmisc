import torch
import torch.nn as nn

from . import args_base as ab
from . import utils as ut


class Ensemble(ab.ArgsBase):

  def __init__(self, router_net, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.router_net = router_net

  def forward(self, *args, **kwargs):
    ry = self.router_net(*args, **kwargs)

    parts = [net(*args, **kwargs) * ry[:, i] for i, net in enumerate(self.values())]
    y = ut.add(*parts) / len(parts)

    return y

