import torch
import torch.nn as nn

from . import utils as ut


class CatLoss(nn.Module):

  def __init__(self, loss):
    super().__init__()
    self.loss = loss

  def forward(self, y, targets):
    if targets is not None:
      # If training or the targets are not integers, call the loss directly.
      if getattr(self.loss, 'training', True) or not ut.is_integer(targets):
        return self.loss(y, targets)

      # If not training the "loss" is meant as the categorization error.
      _, predicted = torch.max(y, dim=1)
      correct = (predicted == targets).sum()

      return 1.0 - correct / y.shape[0]


class SeqLoss(nn.Module):

  def __init__(self, loss):
    super().__init__()
    self.loss = loss

  def forward(self, y, targets):
    if targets is not None:
      # Flatten batch and sequence dimensions together.
      y = y.view(-1, y.shape[-1])

      return self.loss(y, targets.flatten())

