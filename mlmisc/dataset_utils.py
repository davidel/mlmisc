import inspect
import os
import random

import datasets as dsets
import huggingface_hub as hfh
import matplotlib.pyplot as plt
import py_misc_utils.alog as alog
import py_misc_utils.assert_checks as tas
import py_misc_utils.gen_fs as gfs
import py_misc_utils.module_utils as pymu
import py_misc_utils.utils as pyu
import torch
import torchvision

from . import dataset_base as dsb
from . import utils as ut


class Dataset(dsb.Dataset):

  def __init__(self, data,
               select_fn=None,
               transform=None,
               target_transform=None,
               **kwargs):
    super().__init__(select_fn=select_fn,
                     transform=transform,
                     target_transform=target_transform,
                     **kwargs)
    self._data = data

  def extra_arg(self, name):
    for source in (super(), self._data):
      extra_arg = getattr(source, 'extra_arg', None)
      if extra_arg is not None:
        xarg = extra_arg(name)
        if xarg is not None:
          return xarg

    return getattr(self._data, name, None)

  def get_sample(self, i):
    if isinstance(self._data, dict):
      return {k: v[i] for k, v in self._data.items()}

    return self._data[i]

  def __len__(self):
    if isinstance(self._data, dict):
      return min(len(v) if hasattr(v, '__len__') else 1 for v in self._data.values())

    return len(self._data)


class IterableDataset(dsb.IterableDataset):

  def __init__(self, data,
               select_fn=None,
               transform=None,
               target_transform=None,
               **kwargs):
    super().__init__(select_fn=select_fn,
                     transform=transform,
                     target_transform=target_transform,
                     **kwargs)
    self._data = data

  def extra_arg(self, name):
    for source in (super(), self._data):
      extra_arg = getattr(source, 'extra_arg', None)
      if extra_arg is not None:
        xarg = extra_arg(name)
        if xarg is not None:
          return xarg

    return getattr(self._data, name, None)

  def enum_samples(seld):
    for data in self._data:
      yield data


def get_dataset_base(dataset):
  return Dataset if hasattr(dataset, '__getitem__') else IterableDataset


def _get_dataset_path(name, cache_dir, dataset_kwargs):
  ds_path = os.path.join(cache_dir, *name.split('/'))

  if gfs.exists(ds_path):
    alog.debug(f'Dataset "{name}" already cached at {ds_path}')
    download = dataset_kwargs.pop('download', None)
    if download == 'force':
      alog.info(f'Forcing download of dataset "{name}" into {ds_path}')
      gfs.rmtree(ds_path)
      gfs.makedirs(ds_path)
      dataset_kwargs['download'] = True
    elif download is not None:
      dataset_kwargs['download'] = False

  return ds_path


def _try_torchvision(name, ds_path, select_fn, transform, target_transform, split_pct,
                     dataset_kwargs):
  dsclass = getattr(torchvision.datasets, name, None)
  if dsclass is not None:
    sig = inspect.signature(dsclass)
    kwargs = dataset_kwargs.copy()
    if sig.parameters.get('download') is not None and 'download' not in kwargs:
      kwargs.update(download=True)
    ds_seed = kwargs.pop('ds_seed', None)

    ds = dict()
    if sig.parameters.get('train') is not None:
      train_ds = dsclass(root=ds_path, train=True, **kwargs)

      kwargs.pop('download', None)
      test_ds = dsclass(root=ds_path, train=False, **kwargs)
    elif sig.parameters.get('split') is not None:
      train_split = kwargs.pop('train_split', 'train')
      test_split = kwargs.pop('test_split', 'test')

      train_ds = dsclass(root=ds_path, split=train_split, **kwargs)

      kwargs.pop('download', None)
      test_ds = dsclass(root=ds_path, split=test_split, **kwargs)
    else:
      full_ds = dsclass(root=ds_path, **kwargs)

      # If we split the dataset ourselves, we do not know if the distribution of samples
      # is uniform within the dataset, so we shuffle indices and create a randomized one.
      shuffled_indices = dsb.shuffled_indices(len(full_ds), seed=ds_seed)
      ntrain = int(split_pct * len(shuffled_indices))

      alog.info(f'Loading torchvision "{name}" dataset whose API does not support splits. ' \
                f'Shuffling indices and using {split_pct:.2f} split')

      train_ds = dsb.SubDataset(full_ds, shuffled_indices[: ntrain])
      test_ds = dsb.SubDataset(full_ds, shuffled_indices[ntrain:])

    ds_base = get_dataset_base(train_ds)
    ds['train'] = ds_base(train_ds,
                          select_fn=select_fn,
                          transform=transform.get('train'),
                          target_transform=target_transform.get('train'),
                          **kwargs)
    ds['test'] = ds_base(test_ds,
                         select_fn=select_fn,
                         transform=transform.get('test'),
                         target_transform=target_transform.get('test'),
                         **kwargs)

    return ds


def items_selector(items):

  def select_fn(x):
    return [x[i] for i in items]

  return select_fn


def _norm_transforms(transform):
  if isinstance(transform, dict):
    return transform

  return dict(train=transform, test=transform)


def _try_module(name, ds_path, select_fn, transform, target_transform, split_pct,
                dataset_kwargs):
  parts = name.split(':', maxsplit=1)
  if len(parts) == 2:
    modpath, ctor_fn = parts
    try:
      module = pymu.import_module(gfs.normpath(modpath))
    except ImportError:
      return

    ctor = pymu.module_getter(ctor_fn)(module)

    kwargs = pyu.dict_setmissing(dataset_kwargs,
                                 cache_dir=ds_path,
                                 split_pct=split_pct)

    mds = ctor(**kwargs)
    if isinstance(mds, (list, tuple)):
      train_ds, test_ds = mds
    else:
      train_ds, test_ds = mds['train'], mds['test']

    ds = dict()
    ds_base = get_dataset_base(train_ds)
    ds['train'] = ds_base(train_ds,
                          select_fn=select_fn,
                          transform=transform.get('train'),
                          target_transform=target_transform.get('train'),
                          **kwargs)
    ds['test'] = ds_base(test_ds,
                         select_fn=select_fn,
                         transform=transform.get('test'),
                         target_transform=target_transform.get('test'),
                         **kwargs)

    return ds


def create_dataset(name,
                   cache_dir=None,
                   select_fn=None,
                   transform=None,
                   target_transform=None,
                   split_pct=None,
                   dataset_kwargs=None):
  cache_dir = cache_dir or os.path.join(os.getenv('HOME', '.'), 'datasets')
  select_fn = pyu.value_or(select_fn, dsb.ident_select)
  transform = _norm_transforms(transform)
  target_transform = _norm_transforms(target_transform)
  split_pct = pyu.value_or(split_pct, 0.9)
  dataset_kwargs = pyu.value_or(dataset_kwargs, {})

  ds_path = _get_dataset_path(name, cache_dir, dataset_kwargs)

  ds = _try_module(name, ds_path, select_fn, transform, target_transform, split_pct,
                   dataset_kwargs)
  if ds is not None:
    return ds

  if name.find('/') < 0:
    ds = _try_torchvision(name, ds_path, select_fn, transform, target_transform,
                          split_pct, dataset_kwargs)
    if ds is not None:
      return ds

  if name in [dset.id for dset in hfh.list_datasets(dataset_name=name)]:
    hfds = dsets.load_dataset(name, cache_dir=ds_path, **dataset_kwargs)

    ds = dict()
    ds_base = get_dataset_base(hfds['train'])
    ds['train'] = ds_base(hfds['train'],
                          select_fn=select_fn,
                          transform=transform.get('train'),
                          target_transform=target_transform.get('train'),
                          **dataset_kwargs)
    ds['test'] = ds_base(hfds['test'],
                         select_fn=select_fn,
                         transform=transform.get('test'),
                         target_transform=target_transform.get('test'),
                         **dataset_kwargs)

    return ds

  alog.xraise(ValueError, f'Unable to create dataset: "{name}"')


def get_class_weights(data,
                      dtype=None,
                      cdtype=None,
                      output_filter=None,
                      max_samples=None):
  cdtype = pyu.value_or(cdtype, torch.int32)
  tas.check(ut.torch_is_integer_dtype(cdtype),
            msg=f'Targets should be class integers instead of {cdtype}')
  # By default assume target is the second entry in the dataset return tuple.
  output_filter = pyu.value_or(output_filter, lambda x: x[1])

  indices = list(range(len(data)))
  if max_samples is not None and len(indices) > max_samples:
    random.shuffle(indices)
    indices = indices[: max_samples]

  target = torch.empty(len(indices), dtype=cdtype)
  for i, idx in enumerate(indices):
    y = output_filter(data[idx])
    target[i] = ut.item(y)

  cvalues, class_counts = torch.unique(target, return_counts=True)

  weight = 1.0 / class_counts
  weight = weight / torch.sum(weight)
  if dtype is not None:
    weight = weight.to(dtype)

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

