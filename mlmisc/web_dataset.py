import io
import json
import random
import re
import tarfile
import yaml

import msgpack
import numpy as np
import py_misc_utils.alog as alog
import py_misc_utils.img_utils as pyimg
import py_misc_utils.stream_url as pysu
import torch

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
    for k, v in data.items():
      dpos = k.rfind('.')
      fmt = k[dpos + 1:] if dpos >= 0 else k

      if fmt in {'jpg', 'png', 'jpeg'}:
        ddata[k] = pyimg.from_bytes(v)
      elif fmt == 'json':
        ddata[k] = json.loads(v)
      elif fmt in {'pth', 'pt'}:
        ddata[k] = torch.load(io.BytesIO(v), weights_only=True)
      elif fmt in {'npy', 'npz'}:
        ddata[k] = np.load(io.BytesIO(v), allow_pickle=False)
      elif fmt in {'cls', 'cls2', 'index'}:
        ddata[k] = int(v)
      elif fmt in {'yaml', 'yml'}:
        ddata[k] = yaml.safe_load(io.BytesIO(v))
      elif fmt == 'mp':
        ddata[k] = msgpack.unpackb(v)
      else:
        ddata[k] = v

    return ddata

  def generate(self):
    if self._shuffle in (True, None):
      urls = random.sample(self._urls, len(self._urls))
    else:
      urls = self._urls

    for url in urls:
      alog.debug(f'Opening new stream: {url}')
      stream = pysu.StreamUrl(url, **self._kwargs)

      tar = tarfile.open(mode='r|', fileobj=stream)

      ctid, data = None, dict()
      for tinfo in tar:
        dpos = tinfo.name.find('.')
        if dpos > 0:
          tid = tinfo.name[: dpos]
          ext = tinfo.name[dpos + 1:]

          if tid != ctid and data:
            yield self._decode(data, tid)
            data = dict()

          ctid = tid
          data[ext] = tar.extractfile(tinfo).read()

      if data:
        yield self._decode(data, ctid)

  def __iter__(self):
    return iter(self.generate())

  def __len__(self):
    return self._size


def expand_urls(url):
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


def create(url, shuffle=True, split_pct=0.9, total_samples=None, **kwargs):
  urls = expand_urls(url)
  if shuffle:
    urls = random.sample(urls, len(urls))

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
  if wds_train:
    ds['train'] = WebDataset(train_urls, shuffle=shuffle, size=train_size, **kwargs)
  if wds_test:
    ds['test'] = WebDataset(test_urls, shuffle=shuffle, size=test_size, **kwargs)

  return ds

