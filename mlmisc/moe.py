import torch
import torch.nn as nn

from . import layer_utils as lu


class MOE(nn.Module):

  def __init__(self, n, idim, odim, act=None):
    super().__init__()
    self.weight = nn.Parameter(torch.randn(n, odim, idim))
    self.gates = nn.Linear(idim, n)
    self.act = lu.create(act or 'gelu')

  def forward(self, x):
    # x is (B, ID)
    y = torch.einsum('noi,biz->bnoz', self.weight, torch.unsqueeze(x, -1))
    y = y.squeeze(-1) # (B, N, OD)
    y = torch.permute(y, (0, 2, 1)) # (B, OD, N)

    g = self.act(self.gates(x)) # (B, N)

    z = y @ torch.unsqueeze(g, -1) # (B, OD, 1)

    return z.squeeze(-1) # (B, OD)

