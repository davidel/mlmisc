import torch


class NextTokenDataset(torch.utils.data.Dataset):

  def __init__(self, data, block_size, seqlen=1):
    self.data = data
    self.block_size = block_size
    self.seqlen = seqlen

  def __len__(self):
    return max(len(self.data) - self.block_size - self.seqlen, 0)

  def __getitem__(self, i):
    offset = i + self.block_size

    return self.data[i: offset], self.data[offset: offset + self.seqlen]

