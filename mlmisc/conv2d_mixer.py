import torch
import torch.nn as nn

from . import args_parallel as ap


class Conv2dMixer(nn.Module):

  def __init__(self, in_channels, convs_spec):
    super().__init__()
    convs = []
    for chans, ksize in convs_spec:
      convs.append(nn.Conv2d(in_channels, chans, kernel_size=ksize, padding='same'))
    self.convs = ap.ArgsParallel(convs, cat_dim=1)

  def forward(self, x):
    return self.convs(x)

