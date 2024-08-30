import torch
import torch.nn as nn

from . import utils as ut


class ModuleBuilder(nn.Module):

  def __init__(self, shape):
    super().__init__()
    self.shape = tuple(shape)
    self.layers = nn.ModuleList()
    self.input_fns = []
    self.net_args = []
    self.results = []

  def add(self, net, input_fn=None, net_args=None):
    self.shape = ut.net_shape(net, self.shape)
    self.layers.append(net)
    self.input_fns.append(input_fn)
    self.net_args.append(net_args)

    return len(self.layers) - 1

  def fc(self, odim, input_fn=None, **kwargs):
    return self.add(nn.Linear(self.shape[-1], odim, **kwargs),
                    input_fn=input_fn)

  def conv2d(self, odim, input_fn=None, **kwargs):
    return self.add(nn.Conv2d(self.shape[-3], odim, **kwargs),
                    input_fn=input_fn)

  def deconv2d(self, odim, input_fn=None, **kwargs):
    return self.add(nn.ConvTranspose2d(self.shape[-3], odim, **kwargs),
                    input_fn=input_fn)

  def result(self, i):
    return self.results[i]

  def forward(self, x, **kwargs):
    y = x
    for i, net in enumerate(self.layers):
      input_fn = self.input_fns[i]
      xx = y if input_fn is None else input_fn(y)
      net_args, net_kwargs = self.net_args[i], dict()
      if net_args:
        for k in net_args:
          net_kwargs[k] = kwargs.get(k)

      y = net(xx, **net_kwargs)
      self.results.append(y)

    self.results = []

    return y

