import functools
import math
import operator
import re

import numpy as np
import py_misc_utils.alog as alog
import py_misc_utils.core_utils as pycu
import py_misc_utils.gfs as gfs
import py_misc_utils.module_utils as pymu
import py_misc_utils.np_utils as pyn
import py_misc_utils.rnd_utils as pyr
import py_misc_utils.utils as pyu
import torch
import torch.nn as nn


class Training:

  def __init__(self, net, training):
    self._net = net
    self._training = training

  def __enter__(self):
    self._prev_training = self._net.training
    self._net.train(self._training)
    return self

  def __exit__(self, *exc):
    self._net.train(self._prev_training)
    return False


def get_device(kind=None):
  if kind is None:
    kind = 'cuda' if torch.cuda.is_available() else 'cpu'

  return torch.device(kind)


def is_cuda_device(device):
  return device.type == 'cuda'


def randseed(seed):
  rseed = pyr.manual_seed(seed)
  torch.manual_seed(rseed)
  torch.cuda.manual_seed_all(rseed)

  return rseed


def item(v):
  item_fn = getattr(v, 'item', None)

  return v if item_fn is None else item_fn()


def torch_dtype(dtype):
  return getattr(torch, dtype) if isinstance(dtype, str) else dtype


def dtype_for_size(size, ns=torch, signed=False):
  nbits = math.ceil(math.log2(size))
  if signed:
    nbits += 1
    root = 'int'
  else:
    root = 'uint'

  max_nbits = 64
  n = 8
  while n <= max_nbits:
    if nbits <= n:
      return getattr(ns, f'{root}{n}')
    n *= 2

  alog.xraise(ValueError,
              f'Size {size} too big to fit an integer (must fit {max_nbits} bits)')


def torch_is_integer_dtype(dtype):
  return not (dtype.is_floating_point or dtype.is_complex)


def is_integer(t):
  if isinstance(t, int):
    return True
  if isinstance(t, torch.Tensor):
    return torch_is_integer_dtype(t.dtype)
  if isinstance(t, np.ndarray):
    return pyn.is_integer(t.dtype)

  return False


def torch_load(path, **kwargs):
  with gfs.open_local(path, mode='rb') as ptfd:
    return torch.load(ptfd, weights_only=False, **kwargs)


def torch_load_to(dest, path, **kwargs):
  alog.debug(f'Loading tensor data from {path} to tensor/parameter with ' \
             f'shape {tuple(dest.shape)} ...')
  with gfs.open_local(path, mode='rb') as ptfd:
    t = torch.load(ptfd, weights_only=True, **kwargs)
  dest.data.copy_(getattr(t, 'data', t))

  return dest


def net_shape(net, *shapes, device=None, dtype=None, **kwargs):
  with torch.no_grad(), Training(net, False):
    # Add and remove the artificial batch dimension.
    args = []
    for shape in shapes:
      if len(shape) == 2 and isinstance(shape[0], str):
        name, shape = shape
        kwargs[name] = torch.randn((1,) + tuple(shape), dtype=dtype, device=device)
      else:
        args.append(torch.randn((1,) + tuple(shape), dtype=dtype, device=device))

    y = net(*args, **kwargs)

    return tuple(y.shape[1:])


def update_params(src_net, dest_net, tau=0.0):
  with torch.no_grad():
    src_params = {name: sparam for name, sparam in src_net.named_parameters()}
    for name, dparam in dest_net.named_parameters():
      if (sparam := src_params.get(name)) is not None:
        param = dparam * tau + sparam * (1.0 - tau)
        dparam.copy_(param.detach())
      else:
        alog.warning(f'Missing parameter "{name}" {tuple(dparam.shape)} in source net')


def optimizer_params(optimizer):
  params = dict()
  for pg in optimizer.param_groups:
    for param in pg.get('params', ()):
      params[id(param)] = param

  return tuple(params.values())


def clamp_gradients(params, gmax, norm_type=None):
  if norm_type is None:
    torch.nn.utils.clip_grad_value_(params, gmax)
  else:
    torch.nn.utils.clip_grad_norm_(params, gmax, norm_type=norm_type)


def train_step(net, x, y, optimizer,
               scaler=None,
               device=None,
               amp_dtype=None,
               accum_steps=1,
               stepno=1,
               grad_clip=None,
               zero_grad=True):
  if scaler is not None:
    with torch.autocast(device_type=device.type,
                        dtype=amp_dtype or torch.float16):
      _, loss = net(x, targets=y)
  else:
    _, loss = net(x, targets=y)

  bloss = loss if accum_steps == 1 else loss / accum_steps

  if scaler is not None:
    scaler.scale(bloss).backward()
  else:
    bloss.backward()

  if stepno % accum_steps == 0:
    if scaler is not None:
      if grad_clip is not None and grad_clip > 0:
        scaler.unscale_(optimizer)
        clamp_gradients(net.parameters(), grad_clip)

      scaler.step(optimizer)
      scaler.update()
    else:
      if grad_clip is not None and grad_clip > 0:
        clamp_gradients(net.parameters(), grad_clip)

      optimizer.step()

      if zero_grad:
        optimizer.zero_grad()

  return loss


class NoopTbWriter:

  def __getattr__(self, name):
    if name.startswith('add_') or name in {'flush', 'close'}:
      return pycu.noop

    return super().__getattribute__(name)


def create_tb_writer(path, **kwargs):
  # Unfortunately tensorboard depends on Tensorflow, and that creates all sort
  # of issues with GPUs on Colab (Argh! Issues coming from XLA).
  import torch.utils.tensorboard

  alog.debug(f'Creating TB summary writer in {path}')

  return torch.utils.tensorboard.SummaryWriter(path, **kwargs)


def tb_write(tb_writer, name, value, *args, **kwargs):
  if pycu.isdict(value):
    tb_writer.add_scalars(name, value, *args, **kwargs)
  else:
    tb_writer.add_scalar(name, value, *args, **kwargs)


def count_params(net):
  params = 0
  for p in net.parameters():
    params += torch.numel(p)

  return params


def net_memory_size(net):
  size = 0
  for p in net.parameters():
    size += p.element_size() * p.nelement()

  return size


def named_grads(net):
  grads = []
  for name, param in net.named_parameters():
    if param.grad is not None:
      grads.append((name, param.grad))
    else:
      alog.debug0(f'Parameter has no gradient: {name}')

  return tuple(grads)


def freeze_params(net, freeze=None, thaw=None):
  freeze = freeze or ()
  thaw = thaw or ()
  for name, param in net.named_parameters():
    requires_grad = None
    for rx in freeze:
      if re.match(rx, name):
        requires_grad = False
        break
    if requires_grad is None:
      for rx in thaw:
        if re.match(rx, name):
          requires_grad = True
          break

    if requires_grad is not None:
      alog.debug(f'{"Thawing" if requires_grad else "Freezing"} parameter "{name}"')
      param.requires_grad = requires_grad
    else:
      alog.debug(f'Parameter "{name}" left untouched (requires_grad={param.requires_grad})')

  return net


def split_dims(shape, npop):
  return shape[: -npop], *shape[-npop:]


def extra_repr(**kwargs):
  rstr = pyu.stri(kwargs)

  return rstr[1: -1]


def kuni_tensor(*shape, dtype=None, device=None, a=None):
  t = torch.empty(*shape, dtype=dtype, device=device)

  nn.init.kaiming_uniform_(t, a=math.sqrt(5) if a is None else a)

  return t


def kuni_parameter(*args, **kwargs):
  return nn.Parameter(kuni_tensor(*args, **kwargs))


def add(*args):
  return functools.reduce(operator.add, args)


def mul(*args):
  return functools.reduce(operator.mul, args)


def add_dimension(x, dim, size):
  bx = torch.unsqueeze(x, dim)
  shape = list(bx.shape)
  shape[dim] = size

  return torch.broadcast_to(bx, tuple(shape))


def unsqueeze(x, n, mode='left'):
  ux = x
  for _ in range(n):
    ux = torch.unsqueeze(ux, 0 if mode == 'left' else -1)

  return ux

def broadcast_as(x, ref, mode='left'):
  bx = unsqueeze(x, ref.ndim - x.ndim, mode=mode)
  if mode == 'left':
    shape = tuple(ref.shape[: -x.ndim]) + tuple(x.shape)
  else:
    shape = tuple(x.shape) + tuple(ref.shape[x.ndim: ])

  return torch.broadcast_to(bx, shape)


def create_graph(x, path=None, params=None, model=None, format='svg'):
  import torchviz

  if params is None and model is not None:
    params = dict(model.named_parameters())

  dot = torchviz.make_dot(x, params=params)

  if path is not None:
    dot.format = format
    dot.render(path)

  return dot


def get_lr(optimizer, reduce=None):
  lrs = []
  for pgrp in optimizer.param_groups:
    lr = pgrp.get('lr')
    if lr is not None:
      lrs.append(lr)

  if lrs:
    return tuple(lrs) if reduce is False else np.mean(lrs)


def reset_lr(optimizer, lr):
  for pgrp in optimizer.param_groups:
    if 'lr' in pgrp:
      pgrp['lr'] = lr


def minmax_bbox(bbox):
  # Boxes are (N, 4) shapes with (min_x, min_y, xsize, ysize)
  mmbox = torch.clone(bbox)
  mmbox[:, 2] = bbox[:, 0] + bbox[:, 2]
  mmbox[:, 3] = bbox[:, 1] + bbox[:, 3]

  return mmbox


def get_iou(abox, bbox):
  # Boxes are (..., 4) shapes with (min_x, min_y, max_x, max_y)
  gmask = abox > bbox
  mx = torch.where(gmask, abox, bbox)
  mn = torch.where(gmask, bbox, abox)

  # Intersection box (if not empty) mins are max of mins, and intersection
  # box maxes are min of maxes.
  ixmin = mx[..., 0]
  iymin = mx[..., 1]
  ixmax = mn[..., 2]
  iymax = mn[..., 3]

  idx = ixmax - ixmin
  idx = torch.where(idx > 0, idx, 0)

  idy = iymax - iymin
  idy = torch.where(idy > 0, idy, 0)

  # Intersection areas.
  ia = idx * idy

  # The abox and bbox areas.
  aa = (abox[..., 2] - abox[..., 0]) * (abox[..., 3] - abox[..., 1])
  ab = (bbox[..., 2] - bbox[..., 0]) * (bbox[..., 3] - bbox[..., 1])

  return ia / (aa + ab - ia)


def import_model(modname=None, package=None, path=None, model_args=None):
  name_or_path = path or modname

  module = pymu.import_module(name_or_path, modname=modname, package=package)

  return module.create_model(model_args)

