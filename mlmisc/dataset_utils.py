import inspect
import os
import random

import datasets as dsets
import huggingface_hub as hfh
import matplotlib.pyplot as plt
import py_misc_utils.alog as alog
import py_misc_utils.assert_checks as tas
import py_misc_utils.module_utils as pymu
import py_misc_utils.utils as pyu
import torch
import torchvision

from . import utils as ut


class Dataset(torch.utils.data.Dataset):

  def __init__(self, data,
               select_fn=None,
               transform=None,
               target_transform=None,
               **kwargs):
    super().__init__()
    self.data = data
    self.select_fn = select_fn or _guess_select
    self.transform = transform or _no_transform
    self.target_transform = target_transform or _no_transform
    self.kwargs = kwargs

  def extra_arg(self, name):
    xarg = getattr(self.data, 'extra_arg', None)

    return self.kwargs.get(name) if xarg is None else xarg(name)

  def _get(self, i):
    if isinstance(self.data, dict):
      return {k: v[i] for k, v in self.data.items()}

    return self.data[i]

  def __len__(self):
    if isinstance(self.data, dict):
      return min(len(v) if hasattr(v, '__len__') else 1 for v in self.data.values())

    return len(self.data)

  def __getitem__(self, i):
    if isinstance(i, slice):
      return sub_dataset(self, i)

    idata = self._get(i)
    x, y = self.select_fn(idata)

    return self.transform(x), self.target_transform(y)


def _try_torchvision(name, cache_dir, transform, target_transform, split_pct,
                     dataset_kwargs):
  dsclass = getattr(torchvision.datasets, name, None)
  if dsclass is not None:
    sig = inspect.signature(dsclass)
    kwargs = dataset_kwargs.copy()
    if sig.parameters.get('download') is not None and 'download' not in kwargs:
      kwargs.update(download=True)

    ds = dict()
    if sig.parameters.get('train') is not None:
      ds['train'] = dsclass(root=cache_dir,
                            train=True,
                            transform=transform.get('train'),
                            target_transform=target_transform.get('train'),
                            **kwargs)
      ds['test'] = dsclass(root=cache_dir,
                           train=False,
                           transform=transform.get('test'),
                           target_transform=target_transform.get('test'),
                           **kwargs)
    elif sig.parameters.get('split') is not None:
      train_split = kwargs.pop('train_split', 'train')
      test_split = kwargs.pop('test_split', 'test')

      ds['train'] = dsclass(root=cache_dir,
                            split=train_split,
                            transform=transform.get('train'),
                            target_transform=target_transform.get('train'),
                            **kwargs)
      ds['test'] = dsclass(root=cache_dir,
                           split=test_split,
                           transform=transform.get('test'),
                           target_transform=target_transform.get('test'),
                           **kwargs)
    else:
      full_ds = dsclass(root=cache_dir, **kwargs)

      ntrain = int(split_pct * len(full_ds))

      ds['train'] = Dataset(full_ds[: ntrain],
                            transform=transform.get('train'),
                            target_transform=target_transform.get('train'))
      ds['test'] = Dataset(full_ds[ntrain: ],
                           transform=transform.get('test'),
                           target_transform=target_transform.get('test'))

    return ds


def _guess_select(x):
  if isinstance(x, (list, tuple)):
    return x[: 2]
  if isinstance(x, dict):
    return list(x.values())[: 2]

  return x


def _no_transform(x):
  return x


def keys_selector(keys):

  def select_fn(x):
    return [x[k] for k in keys]

  return select_fn


def _norm_transforms(transform):
  if isinstance(transform, dict):
    return transform

  return dict(train=transform, test=transform)


def sub_dataset(ds, dslice):
  indices = torch.arange(*dslice.indices(len(ds)))

  return torch.utils.data.Subset(ds, indices)


def _try_module(name, cache_dir, split_pct, dataset_kwargs):
  parts = name.split(':', maxsplit=1)
  if len(parts) == 2:
    modpath, ctor_fn = parts
    try:
      module = pymu.import_module(modpath)
    except ImportError:
      return

    ctor = getattr(module, ctor_fn)
    kwargs = dataset_kwargs.copy()
    kwargs.update(cache_dir=cache_dir, split_pct=split_pct)

    ds = ctor(**kwargs)

    return dict(train=ds[0], test=ds[1]) if isinstance(ds, (list, tuple)) else ds


def create_dataset(name,
                   cache_dir=None,
                   select_fn=None,
                   transform=None,
                   target_transform=None,
                   split_pct=None,
                   dataset_kwargs=None):
  cache_dir = cache_dir or os.path.join(os.getenv('HOME', '.'), 'datasets')
  transform = _norm_transforms(transform)
  target_transform = _norm_transforms(target_transform)
  split_pct = split_pct or 0.9
  dataset_kwargs = dataset_kwargs or {}

  ds = _try_module(name, cache_dir, split_pct, dataset_kwargs)
  if ds is not None:
    return ds

  if name.find('/') < 0:
    ds = _try_torchvision(name, cache_dir, transform, target_transform, split_pct,
                          dataset_kwargs)
    if ds is not None:
      return ds

  if name in [dset.id for dset in hfh.list_datasets(dataset_name=name)]:
    hfds = dsets.load_dataset(name, cache_dir=cache_dir, **dataset_kwargs)

    ds = dict()
    ds['train'] = Dataset(hfds['train'],
                          select_fn=select_fn,
                          transform=transform.get('train'),
                          target_transform=target_transform.get('train'))
    ds['test'] = Dataset(hfds['test'],
                         select_fn=select_fn,
                         transform=transform.get('test'),
                         target_transform=target_transform.get('test'))

    return ds

  alog.xraise(ValueError, f'Unable to create dataset: "{name}"')


def get_class_weights(data,
                      dtype=None,
                      cdtype=None,
                      output_filter=None,
                      max_samples=None):
  # By default assume target is the second entry in the dataset return tuple.
  output_filter = output_filter or (lambda x: x[1])

  indices = list(range(len(data)))
  if max_samples is not None and len(indices) > max_samples:
    random.shuffle(indices)
    indices = sorted(indices[: max_samples])

  target = torch.empty(len(indices), dtype=cdtype or torch.int32)
  for i in indices:
    y = output_filter(data[i])
    target[i] = ut.item(y)

  cvalues, class_counts = torch.unique(target, return_counts=True)
  weight = 1.0 / class_counts
  weight = weight / torch.sum(weight)

  if dtype is not None:
    weight = weight.to(dtype)

  if ut.is_integer(cvalues):
    max_class = torch.max(cvalues).item()
    if max_class >= len(cvalues):
      fweight = torch.zeros(max_class + 1, dtype=weight.dtype)
      fweight[cvalues] = weight
      weight = fweight

  alog.debug(f'Data class weight: { {c: f"{n:.2e}" for c, n in enumerate(weight)} }')

  return weight


def show_images(dataset, n, path=None):
  for i, iid in enumerate(random.sample(range(len(dataset)), k=n)):
    img, label = dataset[iid]

    tas.check_eq(img.ndim, 3,
                 msg=f'Incorrect shape for image (should be (C, H, W)): {tuple(img.shape)}')

    shimg = torch.permute(img, (1, 2, 0))
    alog.debug(f'Image: shape={tuple(shimg.shape)} label={label}')

    plt.title(f'Label = {label}')
    plt.imshow(shimg, interpolation='bicubic')

    if path is None:
      plt.show()
    else:
      if n > 1:
        fpath, ext = os.path.splitext(path)
        fpath = f'{fpath}.{i}{ext}'
      else:
        fpath = path

      plt.savefig(fpath)

