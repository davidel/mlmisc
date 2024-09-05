import torch
import torch.nn as nn

from . import args_parallel as ap


class Conv2dMixer(nn.Module):

  def __init__(self, in_channels, out_channels, convs_spec):
    super().__init__() -
    convs = []
    for chans, ksize in convs_spec:
      convs.append(nn.Conv2d(in_channels, chans, kernel_size=ksize, padding='same'))
    self.convs = ap.ArgsParallel(convs, cat_dim=1)
    self.conv_proj = nn.Conv2d(sum(s[0] for s in convs_spec), out_channels,
                               kernel_size=1,
                               padding='same')

  def forward(self, x):
    y = self.convs(x)
    y = self.conv_proj(y)

    return y
