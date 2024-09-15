import torch
import torch.nn as nn

from . import args_base as ab
from . import utils as ut


class Ensemble(ab.ArgsBase):

  def __init__(self, router_net, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.nets = tuple(self.values())
    self.router_net = router_net
    self.softmax = nn.Softmax(dim=-1)

  def forward(self, *args, **kwargs):
    ry = self.router_net(*args, **kwargs)
    ry = self.softmax(ry)

    parts = [net(*args, **kwargs) * torch.unsqueeze(ry[:, i], 1) for i, net in enumerate(self.nets)]
    y = ut.add(*parts)

    return y

