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

  def __init__(self, url, shuffle=None, **kwargs):
    super().__init__()
    self._url = url
    self._shuffle = shuffle
    self._kwargs = kwargs
    self._urls = tuple(expand_urls(url))

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

  def enum_urls(self):
    if self._shuffle in (True, None):
      urls = list(self._urls)
      random.shuffle(urls)
    else:
      urls = self._urls

    for url in urls:
      yield url

  def generate(self):
    for url in self.enum_urls():
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


def create(wds_train=None, wds_test=None, **kwargs):
  ds = dict()
  if wds_train:
    ds['train'] = WebDataset(**wds_train)
  if wds_test:
    ds['test'] = WebDataset(**wds_test)

  return ds

