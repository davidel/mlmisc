import io
import json
import os
import random
import re
import requests
import tarfile
import yaml

import msgpack
import numpy as np
import py_misc_utils.alog as alog
import py_misc_utils.img_utils as pyimg
import py_misc_utils.utils as pyu
import torch

from . import dataset_base as dsb


def next_chunk(resp_iter):
  try:
    return memoryview(next(resp_iter))
  except StopIteration:
    pass


class StreamFile:

  def __init__(self, url, auth=None, chunk_size=1024 * 128, **kwargs):
    headers = dict()
    if auth:
      headers['Authorization'] = auth

    self._url = url
    self._auth = auth
    self._chunk_size = chunk_size
    self._response = requests.get(url, headers=headers, stream=True)
    self._response.raise_for_status()
    self._resp_iter = self._response.iter_content(chunk_size=self._chunk_size)
    self._buffer = next_chunk(self._resp_iter)

  def read(self, size):
    iobuf = io.BytesIO()
    while self._buffer is not None and size > 0:
      if size >= len(self._buffer):
        iobuf.write(self._buffer)
        size -= len(self._buffer)
        self._buffer = next_chunk(self._resp_iter)
      else:
        iobuf.write(self._buffer[: size])
        self._buffer = self._buffer[size:]
        break

    return iobuf.getvalue()


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
      stream = StreamFile(url, **self._kwargs)

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

