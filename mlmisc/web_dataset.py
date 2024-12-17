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
import py_misc_utils.gfs as gfs
import py_misc_utils.img_utils as pyimg
import py_misc_utils.utils as pyu
import torch

from . import dataset_base as dsb


class WebDataset(torch.utils.data.IterableDataset):

  def __init__(self, urls, shuffle=None, size=None, **kwargs):
    shuffle = pyu.value_or(shuffle, True)

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
    return iter(self.generate())

  def __len__(self):
    return self._size


def expand_huggingface_urls(url):
  import huggingface_hub as hfhub

  fs = hfhub.HfFileSystem()
  files = [fs.resolve_path(path) for path in fs.glob(url)]

  return tuple(hfhub.hf_hub_url(rfile.repo_id, rfile.path_in_repo, repo_type='dataset')
               for rfile in files)


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


def create(url,
           shuffle=None,
           split_pct=None,
           total_samples=None,
           seed=None,
           shuffle_buffer_size=None,
           **kwargs):
  shuffle = pyu.value_or(shuffle, True)
  split_pct = pyu.value_or(split_pct, 0.9)

  urls = expand_urls(url)
  if shuffle:
    # Stable shuffling, given same seed. Even though the WebDataset (and the
    # ShufflerDataset) do shuffle urls/samples, because of the way we split
    # between train/test urls (by slicing), randomization is needed since the
    # distribution might not be uniform among the dataset urls.
    urls = dsb.shuffled_data(urls, seed=seed)

  ntrain = int(split_pct * len(urls))
  train_urls = urls[: ntrain]
  test_urls = urls[ntrain:]

  if total_samples is not None:
    samples_per_shard = total_samples // len(urls)
    train_size = samples_per_shard * len(train_urls)
    test_size = samples_per_shard * len(test_urls)
  else:
    train_size = test_size = None

  ds = dict()
  ds['train'] = WebDataset(train_urls, shuffle=shuffle, size=train_size, **kwargs)
  ds['test'] = WebDataset(test_urls, shuffle=shuffle, size=test_size, **kwargs)
  if shuffle:
    ds['train'] = dsb.ShufflerDataset(ds['train'], buffer_size=shuffle_buffer_size)
    ds['test'] = dsb.ShufflerDataset(ds['test'], buffer_size=shuffle_buffer_size)

  return ds

