import torch
import torch.nn as nn


class CatLoss(nn.Module):

  def __init__(self, loss):
    super().__init__()
    self.loss = loss

  def forward(self, y, targets):
    if targets is not None:
      if getattr(self.loss, 'training', True):
        return self.loss(y, targets)

      _, predicted = torch.max(y, dim=1)
      correct = (predicted == targets).sum()

      return 1.0 - correct / y.shape[0]


class SeqLoss(nn.Module):

  def __init__(self, loss):
    super().__init__()
    self.loss = loss

  def forward(self, y, targets):
    if targets is not None:
      # Flatten batch and sequence dimension together.
      y = y.view(-1, y.shape[-1])

      return self.loss(y, targets.flatten())

