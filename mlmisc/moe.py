import einops
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
    # (B, IN) => (B, IN, 1)
    xx = torch.unsqueeze(x, -1)
    # (N, OUT, IN) @ (B, IN, 1) => (B, N, OUT, 1)
    y = torch.einsum('noi,biz->bnoz', self.weight, xx)
    # (B, N, OUT, 1) => (B, OUT, N)
    y = einops.rearrange(y, 'b n o z -> b (o z) n')
    # (B, IN) => (B, N)
    g = self.act(self.gates(x))
    # (B, N) => (B, N, 1)
    g = torch.unsqueeze(g, -1)
    # (B, OUT, N) @ (B, N, 1) => (B, OUT, 1)
    y = y @ g
    # (B, OUT, 1) => (B, OUT)
    y = torch.squeeze(y, -1)

    return y

