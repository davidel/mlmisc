import torch
import torch.nn.functional as F


class NextTokenDataset(torch.utils.data.Dataset):

  def __init__(self, data, block_size, seqlen=1, pad=None):
    self.data = data
    self.block_size = block_size
    self.seqlen = seqlen
    self.pad = pad

  def __len__(self):
    return max(len(self.data) - self.block_size - self.seqlen, 0)

  def __getitem__(self, i):
    offset = i + self.block_size
    x, y = self.data[i: offset], self.data[offset: offset + self.seqlen]

    if self.pad is not None:
      x = F.pad(x, self.pad['pad'], value=self.pad['value'])

    return x, y

