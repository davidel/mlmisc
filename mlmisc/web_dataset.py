import io
import json
import os
import random
import re
import yaml

import msgpack
import numpy as np
import py_misc_utils.alog as alog
import py_misc_utils.archive_streamer as pyas
import py_misc_utils.core_utils as pycu
import py_misc_utils.data_cache as pydc
import py_misc_utils.gfs as gfs
import py_misc_utils.img_utils as pyimg
import py_misc_utils.utils as pyu
import torch

from . import dataset_adapters as dsad
from . import dataset_base as dsb


class WebDataset(torch.utils.data.IterableDataset):

  def __init__(self, urls, shuffle=True, size=None, **kwargs):
    super().__init__()
    self._shuffle = shuffle
    self._size = size
    self._kwargs = kwargs
    self._urls = tuple(expand_urls(urls)) if isinstance(urls, str) else tuple(urls)

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


def expand_huggingface_urls(url, cache_dir=None):
  import huggingface_hub as hfhub

  with pydc.DataCache(url, cache_dir=cache_dir, max_age=28800) as dc:
    if (hf_files := dc.data()) is None:
      fs = hfhub.HfFileSystem()
      files = [fs.resolve_path(path) for path in fs.glob(url)]
      hf_files = tuple(hfhub.hf_hub_url(rfile.repo_id, rfile.path_in_repo, repo_type='dataset')
                       for rfile in files)

      dc.store(hf_files)

    return hf_files


def expand_urls(url):
  if gfs.get_proto(url) == 'hf':
    return expand_huggingface_urls(url)
  else:
    m = re.match(r'(.*)\{(\d+)\.\.(\d+)\}(.*)', url)
    if m:
      isize = len(m.group(2))
      start = int(m.group(2))
      end = int(m.group(3))
      urls = []
      for i in range(start, end + 1):
        urls.append(m.group(1) + f'{i:0{isize}d}' + m.group(4))

      return urls

    return [url]


def expand_dataset_urls(dsinfo, shuffle=True, seed=None):
  train = test = None
  if pycu.isdict(dsinfo):
    train = expand_urls(dsinfo['train'])
    test = expand_urls(dsinfo['test'])
  else:
    train = expand_urls(dsinfo)

  if shuffle:
    # Stable shuffling, given same seed. Even though the WebDataset (and the
    # ShufflerDataset) do shuffle urls/samples, because of the way we split
    # between train/test urls (by slicing), randomization is needed since the
    # distribution might not be uniform among the dataset urls.
    if train is not None:
      train = dsb.shuffled_data(train, seed=seed)
    if test is not None:
      test = dsb.shuffled_data(test, seed=seed)

  return pyu.make_object(train=train, test=test)


def create(url,
           url_shuffle=True,
           shuffle=True,
           split_pct=0.9,
           train_size=None,
           test_size=None,
           seed=None,
           shuffle_buffer_size=1024,
           **kwargs):
  ds_urls = expand_dataset_urls(url, shuffle=url_shuffle, seed=seed)

  if ds_urls.test is None:
    ntrain = int(split_pct * len(ds_urls.train))
    train_urls = ds_urls.train[: ntrain]
    test_urls = ds_urls.train[ntrain:]
  else:
    train_urls = ds_urls.train
    test_urls = ds_urls.test

  ds = dict()
  ds['train'] = WebDataset(train_urls, shuffle=shuffle, size=train_size, **kwargs)
  ds['test'] = WebDataset(test_urls, shuffle=shuffle, size=test_size, **kwargs)
  if shuffle:
    ds['train'] = dsad.ShufflerDataset(ds['train'], buffer_size=shuffle_buffer_size)
    ds['test'] = dsad.ShufflerDataset(ds['test'], buffer_size=shuffle_buffer_size)

  return ds

