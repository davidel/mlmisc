import torch


class NextTokenDataset(torch.utils.data.Dataset):

  def __init__(self, data, block_size):
    self.data = data
    self.block_size = block_size

  def __len__(self):
    return len(self.data) - self.block_size

  def __getitem__(self, i):
    offset = i + self.block_size

    return self.data[i: offset], self.data[offset: offset + 1]

