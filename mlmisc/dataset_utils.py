import inspect
import os
import random

import datasets as dsets
import huggingface_hub as hfh
import matplotlib.pyplot as plt
import py_misc_utils.alog as alog
import py_misc_utils.assert_checks as tas
import py_misc_utils.fs_utils as pyfsu
import py_misc_utils.gfs as gfs
import py_misc_utils.inspect_utils as pyiu
import py_misc_utils.module_utils as pymu
import py_misc_utils.pipeline as pypl
import py_misc_utils.utils as pyu
import torch
import torchvision

from . import core_utils as cu
from . import dataset_base as dsb
from . import utils as ut


class Dataset(dsb.Dataset):

  def __init__(self, data, pipeline=None, **kwargs):
    super().__init__(pipeline=pipeline, **kwargs)
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

  def __init__(self, data, pipeline=None, **kwargs):
    super().__init__(pipeline=pipeline, **kwargs)
    self._data = data

  def extra_arg(self, name):
    for source in (super(), self._data):
      extra_arg = getattr(source, 'extra_arg', None)
      if extra_arg is not None:
        xarg = extra_arg(name)
        if xarg is not None:
          return xarg

    return getattr(self._data, name, None)

  def enum_samples(self):
    yield from self._data

  def __len__(self):
    dslen = dataset_size(self._data)

    return dslen if dslen is not None else dataset_size(super())


def is_random_access_dataset(dataset):
  if isinstance(dataset, (torch.utils.data.IterableDataset, dsets.IterableDataset)):
    return False

  return hasattr(dataset, '__getitem__') and hasattr(dataset, '__len__')


def dataset_size(dataset):
  dslen = getattr(dataset, '__len__', None)

  return dslen() if dslen is not None else None


def _get_dataset_base(dataset):
  return Dataset if is_random_access_dataset(dataset) else IterableDataset


def _build_dataset_dict(train_ds, test_ds, pipelines, dataset_kwargs,
                        train_kwargs, test_kwargs):
  ds_base = _get_dataset_base(train_ds)
  ds = dict()
  kwargs = dataset_kwargs.copy()
  kwargs.update(train_kwargs)
  ds['train'] = ds_base(train_ds, pipeline=pipelines.train, **kwargs)

  kwargs = dataset_kwargs.copy()
  kwargs.update(test_kwargs)
  ds['test'] = ds_base(test_ds, pipeline=pipelines.test, **kwargs)

  return ds


def _get_dataset_path(name, cache_dir, dataset_kwargs):
  sname = name.replace(':', '_')
  ds_path = os.path.join(cache_dir, *sname.split('/'))

  if gfs.exists(ds_path):
    alog.debug0(f'Dataset "{name}" already cached at {ds_path}')
    download = dataset_kwargs.pop('download', None)
    if download == 'force':
      alog.info(f'Forcing download of dataset "{name}" into {ds_path}')
      gfs.rmtree(ds_path)
      gfs.makedirs(ds_path)
      dataset_kwargs['download'] = True
    elif download is not None:
      dataset_kwargs['download'] = False

  return ds_path


def _try_torchvision(name, ds_path, split_pct, dataset_kwargs):
  dsclass = getattr(torchvision.datasets, name, None)
  if dsclass is not None:
    dataset_kwargs = dataset_kwargs.copy()
    dataset_kwargs.update(root=ds_path)

    args, kwargs = pyiu.fetch_args(dsclass, dataset_kwargs)

    if 'train' in kwargs:
      kwargs.update(train=True)
      train_ds = dsclass(*args, **kwargs)

      kwargs.pop('download', None)
      kwargs.update(train=False)
      test_ds = dsclass(*args, **kwargs)
    elif 'split' in kwargs:
      train_split = dataset_kwargs.get('train_split', 'train')
      test_split = dataset_kwargs.get('test_split', 'test')

      kwargs.update(split=train_split)
      train_ds = dsclass(*args, **kwargs)

      kwargs.pop('download', None)
      kwargs.update(split=test_split)
      test_ds = dsclass(*args, **kwargs)
    else:
      full_ds = dsclass(*args, **kwargs)

      # If we split the dataset ourselves, we do not know if the distribution of samples
      # is uniform within the dataset, so we shuffle indices and create a randomized one.
      shuffled_indices = dsb.shuffled_indices(len(full_ds),
                                              seed=dataset_kwargs.get('ds_seed'))
      ntrain = int(split_pct * len(shuffled_indices))

      alog.info(f'Loading torchvision "{name}" dataset whose API does not support splits. ' \
                f'Shuffling indices and using {split_pct:.2f} split')

      train_ds = dsb.SubDataset(full_ds, shuffled_indices[: ntrain])
      test_ds = dsb.SubDataset(full_ds, shuffled_indices[ntrain:])

    return dict(train=train_ds, test=test_ds)


def build_pipelines(select_fn=None,
                    transform=None,
                    target_transform=None):
  train, test = pypl.Pipeline(), pypl.Pipeline()

  if select_fn is not None:
    train.add(select_fn)
    test.add(select_fn)

  train_transform = train_target_transform = None
  test_transform = test_target_transform = None
  if transform:
    if isinstance(transform, dict):
      train_transform = transform['train']
      test_transform = transform['test']
    else:
      train_transform = test_transform = transform

  if target_transform:
    if isinstance(target_transform, dict):
      train_target_transform = target_transform['train']
      test_target_transform = target_transform['test']
    else:
      train_target_transform = test_target_transform = target_transform

  if train_transform or train_target_transform:
    train.add(dsb.transformer(sample=train_transform, target=train_target_transform))
  if test_transform or test_target_transform:
    test.add(dsb.transformer(sample=test_transform, target=test_target_transform))

  return pyu.make_object(train=train, test=test)


def _try_module(name, ds_path, split_pct, dataset_kwargs):
  parts = name.split(':', maxsplit=1)
  if len(parts) == 2:
    modpath, ctor_fn = parts
    try:
      module = pymu.import_module(modpath)
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

    return dict(train=train_ds, test=test_ds)


def create_dataset(name,
                   cache_dir=None,
                   select_fn=None,
                   transform=None,
                   target_transform=None,
                   split_pct=0.9,
                   dataset_kwargs=None):
  cache_dir = cache_dir or os.path.join(pyfsu.home(), 'datasets')
  dataset_kwargs = pyu.value_or(dataset_kwargs, {})

  ds_path = _get_dataset_path(name, cache_dir, dataset_kwargs)

  train_kwargs = dataset_kwargs.pop('train', dict())
  test_kwargs = dataset_kwargs.pop('test', dict())

  ds = _try_module(name, ds_path, split_pct, dataset_kwargs)
  if ds is None and name.find('/') < 0:
    ds = _try_torchvision(name, ds_path, split_pct, dataset_kwargs)

  if ds is None and name in [dset.id for dset in hfh.list_datasets(dataset_name=name)]:
    ds = dsets.load_dataset(name, cache_dir=ds_path, **dataset_kwargs)

    if (not is_random_access_dataset(ds['train']) and
        (sbsize := train_kwargs.get('shuffle', 0)) > 0):
      ds['train'] = ds['train'].shuffle(buffer_size=sbsize)
    if (not is_random_access_dataset(ds['test']) and
        (sbsize := test_kwargs.get('shuffle', 0)) > 0):
      ds['test'] = ds['test'].shuffle(buffer_size=sbsize)

  if ds is None:
    alog.xraise(ValueError, f'Unable to create dataset: "{name}"')

  pipelines = build_pipelines(select_fn=select_fn,
                              transform=transform,
                              target_transform=target_transform)

  return _build_dataset_dict(ds['train'], ds['test'], pipelines, dataset_kwargs,
                             train_kwargs, test_kwargs)


def get_class_weights(data,
                      dtype=None,
                      cdtype=torch.int32,
                      output_filter=lambda x: x[1],
                      max_samples=None):
  tas.check(cu.torch_is_integer_dtype(cdtype),
            msg=f'Targets should be class integers instead of {cdtype}')

  indices = list(range(len(data)))
  if max_samples is not None and len(indices) > max_samples:
    random.shuffle(indices)
    indices = indices[: max_samples]

  target = torch.empty(len(indices), dtype=cdtype)
  for i, idx in enumerate(indices):
    y = output_filter(data[idx])
    target[i] = cu.item(y)

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

  alog.debug0(f'Data class weight: { {c: f"{n:.2e}" for c, n in enumerate(weight)} }')

  return weight


def show_images(dataset, n, path=None):
  for i, iid in enumerate(random.sample(range(len(dataset)), k=n)):
    img, label = dataset[iid]

    tas.check_eq(img.ndim, 3,
                 msg=f'Incorrect shape for image (should be (C, H, W)): {tuple(img.shape)}')

    shimg = torch.permute(img, (1, 2, 0))
    alog.debug0(f'Image: shape={tuple(shimg.shape)} label={label}')

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

