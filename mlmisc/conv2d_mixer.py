import torch
import torch.nn as nn

from . import args_parallel as ap
from . import layer_utils as lu


class Conv2dMixer(nn.Module):

  def __init__(self, in_channels, out_channels, convs_spec, act=None):
    act = act or nn.Identity
    convs, total_channels = [], 0
    for chans, ksize in convs_spec:
      convs.append(nn.Conv2d(in_channels, chans, kernel_size=ksize, padding='same'))
      total_channels += ksize

    super().__init__()
    self.act = lu.create(act)
    self.convs = ap.ArgsParallel(convs, cat_dim=1)
    self.conv_proj = nn.Conv2d(total_channels, out_channels, kernel_size=1)

  def forward(self, x):
    y = self.convs(x)
    y = self.act(y)
    y = self.conv_proj(y)

    return y

