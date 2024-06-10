import torch
import torch.nn as nn


class NetBase(nn.Module):

  def __init__(self, device=None):
    super().__init__()
    self.device = device or torch.device('cpu')

  def to(self, *args, **kwargs):
    if args and isinstance(args[0], (str, torch.device)):
      device = torch.device(args[0])
    else if 'device' in kwargs:
      device = torch.device(kwargs['device'])
    else:
      device = self.device

    result =  super().to(*args, **kwargs)

    self.device = device

    return result

