import torch
import torch.nn as nn


class Conv2dMixer(nn.Module):

  def __init__(self, in_channels, convs_spec):
    super().__init__()
    self.convs = nn.ModuleList()
    for chans, ksize in convs_spec:
      self.convs.append(nn.Conv2d(in_channels, chans, kernel_size=ksize, padding='same'))

  def forward(self, x):
    return torch.cat([net(x) for net in self.convs], dim=1)

