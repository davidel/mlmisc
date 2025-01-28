import collections
import functools
import itertools

import py_misc_utils.alog as alog
import py_misc_utils.assert_checks as tas
import py_misc_utils.utils as pyu
import torch
import torch.nn as nn

from ... import conv_utils as cu
from ... import layer_utils as lu
from ... import loss_wrappers as lsw
from ... import module_builder as mb
from ... import net_base as nb


def _compute_sizes(size, num_channels, paddings):
  ipad = iter(paddings)
  csize = size
  for i in range(num_channels):
    csize = cu.conv_wndsize(csize, 3, 1, padding=next(ipad))
    csize = cu.conv_wndsize(csize, 5, 1, padding=next(ipad))
    csize = cu.conv_wndsize(csize, 5, 2)

  enc_size = csize

  ipad = iter(reversed(paddings))
  for i in range(num_channels):
    csize = cu.deconv_wndsize(csize, 5, 2)
    csize = cu.deconv_wndsize(csize, 5, 1, padding=next(ipad))
    csize = cu.deconv_wndsize(csize, 3, 1, padding=next(ipad))

  return csize, enc_size


def _select_padding(paddings):
  spads = [(sum(padding), padding) for padding in paddings]
  spads = sorted(spads, key=lambda x: x[0])

  return spads[0][1]


def _resolve(size, num_channels):
  padding_choices = ((0, 1), (0, 1, 2),) * num_channels
  matches = collections.defaultdict(list)
  for paddings in itertools.product(*padding_choices):
    csize, enc_size = _compute_sizes(size, num_channels, paddings)

    if enc_size > 0 and csize == size:
      matches[enc_size].append(paddings)

  return {enc_size: _select_padding(paddings)
          for enc_size, paddings in matches.items()}


def _net_batchnorm(net, **kwargs):
  net.batchnorm2d(**kwargs)

def _net_layernorm(net, **kwargs):
  net.layernorm(ndims=3, **kwargs)

def _net_nonorm(net, **kwargs):
  pass

_NORMS = {
  'batch': _net_batchnorm,
  'layer': _net_layernorm,
  'none': _net_nonorm,
}

def _get_normfn(norm):
  parts = norm.split(':', maxsplit=1)
  if len(parts) > 1:
    args, kwargs = pyu.parse_args(parts[1])
  else:
    args, kwargs = (), dict()

  return functools.partial(_NORMS[parts[0]], *args, **kwargs)


class Encoder(nb.NetBase):

  def __init__(self, shape, channels,
               paddings=None,
               norm='batch',
               act='relu'):
    ipad = iter(paddings) if paddings is not None else itertools.repeat(0)
    norm_fn = _get_normfn(norm)

    super().__init__()
    self.net = mb.ModuleBuilder(shape)

    self.net.batchnorm2d()

    for nchan in channels[1:]:
      self.net.conv2d(nchan, 3, bias=False, padding=next(ipad))
      norm_fn(self.net)
      self.net.add(lu.create(act))

      self.net.conv2d(nchan, 5, bias=False, padding=next(ipad))
      norm_fn(self.net)
      self.net.add(lu.create(act))

      self.net.conv2d(nchan, 5, stride=2, bias=False)
      norm_fn(self.net)
      self.net.add(lu.create(act))

  def forward(self, x):
    return self.net(x)


class Decoder(nb.NetBase):

  def __init__(self, shape, channels,
               paddings=None,
               norm='batch',
               act='relu'):
    ipad = iter(reversed(paddings)) if paddings is not None else itertools.repeat(0)
    norm_fn = _get_normfn(norm)

    super().__init__()
    self.net = mb.ModuleBuilder(shape)

    for nchan in reversed(channels[: -1]):
      self.net.deconv2d(nchan, 5, stride=2, bias=False)
      norm_fn(self.net)
      self.net.add(lu.create(act))

      self.net.deconv2d(nchan, 5, bias=False, padding=next(ipad))
      norm_fn(self.net)
      self.net.add(lu.create(act))

      self.net.deconv2d(nchan, 3, bias=False, padding=next(ipad))
      if nchan != channels[0]:
        norm_fn(self.net)
        self.net.add(lu.create(act))

  def forward(self, x):
    return self.net(x)


class EncDec(nb.NetBase):

  def __init__(self, shape, channels,
               norm='batch',
               act='relu'):
    w_res = _resolve(shape[-1], len(channels))
    tas.check(w_res, msg=f'Invalid size: {shape[-1]}')
    if shape[-1] != shape[-2]:
      h_res = _resolve(shape[-2], len(channels))
      tas.check(h_res, msg=f'Invalid size: {shape[-2]}')
    else:
      h_res = w_res

    for enc_size in _VALID_FINALS:
      h_paddings = h_res.get(enc_size)
      w_paddings = w_res.get(enc_size)
      if h_paddings and w_paddings:
        break

    tas.check(h_paddings and w_paddings,
              msg=f'Unable to find solution: shape={shape} channels={channels}')

    paddings = tuple(zip(h_paddings, w_paddings))
    all_channels = (shape[0],) + tuple(channels)

    super().__init__()
    self.loss = lsw.Loss(nn.MSELoss())
    self.enc = Encoder(shape, all_channels,
                       paddings=paddings,
                       norm=norm,
                       act=act)
    self.dec = Decoder(self.enc.net.shape, all_channels,
                       paddings=paddings,
                       norm=norm,
                       act=act)

    alog.info(f'Encoder shape: {self.enc.net.shape}')

  def forward(self, x, targets=None):
    z = self.enc(x)
    y = self.dec(z)

    return y, self.loss(y, targets)


def _find_config(size, num_channels, max_span=100):
  for i in range(max_span):
    csize = size + i
    pad = _resolve(csize, num_channels)
    if pad:
      return csize, pad

    csize = size - i
    if i > 0 and csize > 0:
      pad = _resolve(csize, num_channels)
      if pad:
        return csize, pad


Config = collections.namedtuple('Config', 'shape, enc_size, hpad, wpad')

_VALID_FINALS = (2, 1, 3, 4, 5)

def find_config(shape, num_channels,
                max_span=100,
                valid_finals=_VALID_FINALS):
  wres = _find_config(shape[-1], num_channels, max_span=max_span)
  if wres:
    wsize, wpad = wres

    if shape[-1] != shape[-2]:
      hres = _find_config(shape[-2], num_channels, max_span=max_span)
    else:
      hres = wres

    if hres:
      hsize, hpad = hres

      for henc in valid_finals:
        h_pad = hpad.get(henc)
        if h_pad:
          for wenc in valid_finals:
            w_pad = wpad.get(wenc)
            if w_pad:
              return Config(shape=(shape[0], hsize, wsize),
                            enc_size=(henc, wenc),
                            hpad=h_pad,
                            wpad=w_pad)


if __name__ == '__main__':
  import py_misc_utils.app_main as app_main

  @app_main.Main
  def main(shape: tuple,
           num_channels,
           max_span=100,
           valid_finals=_VALID_FINALS):
    conf = find_config(shape, num_channels,
                       max_span=max_span,
                       valid_finals=valid_finals)

    if conf:
      alog.info(f'Found Config: {conf}')


  app_main.basic_main(main, description='EncDec Config Find')

