import torch
import torch.nn as nn
import torch.nn.functional as F


class NetBase(nn.Module):

  def __init__(self, device=None):
    super().__init__()
    self.device = device or torch.device('cpu')

  def to(self, *args, **kwargs):
    if args and isinstance(args[0], (str, torch.device)):
      self.device = torch.device(args[0])

    return super().to(*args, **kwargs)


class Dense(nn.Module):
  def __init__(self, nin, nout, act=None, bn=None, **lin_args):
    super().__init__()
    self.lin = nn.Linear(nin, nout, **lin_args)
    self.act = act
    self.bn = bn

  def forward(self, x):
    y = self.lin(x)
    if self.bn is not None:
      y = self.bn(y)
    if self.act is not None:
      y = self.act(y)

    return y


class Conv2d(nn.Module):
  def __init__(self, fin, fout, ksize, act=None, bn=None, **conv_args):
    super().__init__()
    self.conv = nn.Conv2d(fin, fout, ksize, **conv_args)
    self.act = act
    self.bn = bn

  def forward(self, x):
    y = self.conv(x)
    if self.bn is not None:
      y = self.bn(y)
    if self.act is not None:
      y = self.act(y)

    return y


class ConvTranspose2d(nn.Module):
  def __init__(self, fin, fout, ksize, act=None, bn=None, **conv_args):
    super().__init__()
    self.conv = nn.ConvTranspose2d(fin, fout, ksize, **conv_args)
    self.act = act
    self.bn = bn

  def forward(self, x):
    y = self.conv(x)
    if self.bn is not None:
      y = self.bn(y)
    if self.act is not None:
      y = self.act(y)

    return y


class ConvStack(nn.Module):
  def __init__(self, fin, convs, convfn):
    super().__init__()
    self.convs = []
    nf = fin
    for convp in convs:
      convp = convp.copy()
      feats = convp.pop('feats')
      ksize = convp.pop('ksize')
      cconv = convfn(nf, feats, ksize, **convp)
      self.convs.append(cconv)
      self.add_module(f'Conv{len(self.convs) - 1}', cconv)

      nf = feats

  def forward(self, x, xs=None):
    xs = xs or dict()
    ys = []
    y = x
    for i, conv in enumerate(self.convs):
      shy = xs.get(i, None)
      if shy is not None:
        y = y + shy
      y = conv(y)
      ys.append(y)

    return ys


class EncoderStack(nn.Module):
  def __init__(self, count, features, nhead=1, batch_first=True):
    super().__init__()

    self._encoders = []
    for i in range(count):
      enc = nn.TransformerEncoderLayer(features, nhead, batch_first=batch_first)
      self.add_module(f'Enc{i}', enc)
      self._encoders.append(enc)

  def forward(self, x):
    ys = [enc(x) for enc in self._encoders]
    y = torch.cat(ys, dim=-1)

    return y
