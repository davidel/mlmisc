import io
import json
import os
import random
import yaml

import msgpack
import numpy as np
import py_misc_utils.alog as alog
import py_misc_utils.archive_streamer as pyas
import py_misc_utils.gfs as gfs
import py_misc_utils.img_utils as pyimg
import py_misc_utils.pipeline as pypl
import torch

from . import dataset_adapters as dsad
from . import dataset_base as dsb
from . import dataset_utils as dsu


class WebDataset(torch.utils.data.IterableDataset):

  def __init__(self, urls, shuffle=True, size=None, **kwargs):
    super().__init__()
    self._shuffle = shuffle
    self._size = size
    self._kwargs = kwargs
    self._urls = tuple(dsu.expand_urls(urls)) if isinstance(urls, str) else tuple(urls)

  def _decode(self, data, tid):
    ddata = dict()
    ddata['__key__'] = tid
    for name, value in data.items():
      dpos = name.rfind('.')
      if dpos > 0:
        dname, fmt = name[: dpos], name[dpos + 1:]
      else:
        dname = fmt = name

      if fmt in {'jpg', 'png', 'jpeg', 'img'}:
        ddata[dname] = pyimg.from_bytes(value)
      elif fmt == 'json':
        ddata[dname] = json.loads(value)
      elif fmt in {'pth', 'pt', 'pyd'}:
        ddata[dname] = torch.load(io.BytesIO(value), weights_only=True)
      elif fmt in {'npy', 'npz'}:
        ddata[dname] = np.load(io.BytesIO(value), allow_pickle=False)
      elif fmt in {'cls', 'cls2', 'index'}:
        ddata[dname] = int(value)
      elif fmt in {'yaml', 'yml'}:
        ddata[dname] = yaml.safe_load(io.BytesIO(value))
      elif fmt in {'mp', 'msgpack'}:
        ddata[dname] = msgpack.unpackb(value)
      else:
        ddata[dname] = value

    return ddata

  def generate(self):
    if self._shuffle:
      urls = random.sample(self._urls, len(self._urls))
    else:
      urls = self._urls

    for url in urls:
      alog.debug(f'Opening new stream: {url}')

      ars = pyas.ArchiveStreamer(url, **self._kwargs)
      ctid, data = None, dict()
      for ae in ars:
        dpos = ae.name.find('.')
        if dpos > 0:
          tid = ae.name[: dpos]
          name = ae.name[dpos + 1:]

          if tid != ctid and data:
            yield self._decode(data, ctid)
            data = dict()

          ctid = tid
          data[name] = ae.data

      if data:
        yield self._decode(data, ctid)

  def __iter__(self):
    return self.generate()

  def __len__(self):
    return self._size


def create(url,
           url_shuffle=True,
           shuffle=True,
           train_pct=0.9,
           train_size=None,
           test_size=None,
           seed=None,
           shuffle_buffer_size=1024,
           **kwargs):
  ds_urls = dsu.expand_dataset_urls(url, shuffle=url_shuffle, seed=seed)

  if ds_urls.test is None:
    ntrain = int(train_pct * len(ds_urls.train))
    train_urls = ds_urls.train[: ntrain]
    test_urls = ds_urls.train[ntrain:]
  else:
    train_urls = ds_urls.train
    test_urls = ds_urls.test

  ds = dict()
  ds['train'] = WebDataset(train_urls, shuffle=shuffle, size=train_size, **kwargs)
  ds['test'] = WebDataset(test_urls, shuffle=shuffle, size=test_size, **kwargs)
  if shuffle:
    pipeline = pypl.Pipeline(
      dsad.ShuffleProcessor(buffer_size=shuffle_buffer_size),
    )

    ds['train'] = dsad.IterableTransformDataset(ds['train'], pipeline)
    ds['test'] = dsad.IterableTransformDataset(ds['test'], pipeline)

  return ds

