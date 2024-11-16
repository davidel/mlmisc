import collections
import io
import json
import os
import random
import re
import requests
import tarfile
import threading
import yaml

import msgpack
import numpy as np
import py_misc_utils.alog as alog
import py_misc_utils.img_utils as pyimg
import py_misc_utils.utils as pyu
import torch

from . import dataset_base as dsb
from . import utils as ut


class StreamFile:

  def __init__(self, url, auth=None, chunk_size=1024 * 128, **kwargs):
    self._url = url
    self._auth = auth
    self._chunk_size = chunk_size
    self._lock = threading.Lock()
    self._rcond = threading.Condition(self._lock)
    self._buffers = collections.deque()
    self._closed = False

    headers = dict()
    if auth:
      headers['Authorization'] = auth

    self._response = requests.get(url, headers=headers, stream=True)
    self._response.raise_for_status()
    self._exception = None

    self._thread = threading.Thread(target=self._feed)
    self._thread.start()

  def _feed(self):
    exception = None
    try:
      for chunk in self._response.iter_content(chunk_size=self._chunk_size):
        with self._lock:
          self._buffers.append(memoryview(chunk))
          self._rcond.notify()

          if self._closed:
            break
    except Exception as ex:
      alog.warning(f'While reading HTTP content: {ex}')
      exception = ex

    with self._lock:
      self._closed = True
      self._exception = exception
      self._rcond.notify()

  def close(self):
    with self._lock:
      self._closed = True
      self._rcond.notify()

    if self._thread is not None:
      self._thread.join()
      self._thread = None

  def wait_data(self):
    while not self._buffers and not self._closed:
      self._rcond.wait()

    if self._exception:
      raise self._exception

    return len(self._buffers) > 0

  def read(self, size):
    iobuf = io.BytesIO()
    while size > 0:
      with self._lock:
        if not self.wait_data():
          break

        buf = self._buffers[0]
        if size >= len(buf):
          iobuf.write(buf)

          self._buffers.popleft()

          size -= len(buf)
        else:
          iobuf.write(buf[: size])
          self._buffers[0] = buf[size:]
          size = 0

        del buf

    return iobuf.getvalue()


class WebDataset(torch.utils.data.IterableDataset):

  def __init__(self, url,
               shuffle=None,
               **kwargs):
    files = expand_files(url)
    if shuffle in (True, None):
      random.shuffle(files)

    super().__init__()
    self._kwargs = kwargs
    self._files = tuple(files)

  def _decode(self, data, tid):
    ddata = dict()
    ddata['__key__'] = tid
    for k, v in data.items():
      if k in {'jpg', 'png', 'jpeg'}:
        ddata[k] = pyimg.from_bytes(v)
      elif k == 'json':
        ddata[k] = json.loads(v)
      elif k in {'pth', 'pt'}:
        ddata[k] = torch.load(io.BytesIO(v), weights_only=True)
      elif k in {'npy', 'npz'}:
        ddata[k] = np.load(io.BytesIO(v), allow_pickle=False)
      elif k in {'cls', 'cls2', 'index'}:
        ddata[k] = int(v)
      elif k in {'yaml', 'yml'}:
        ddata[k] = yaml.safe_load(io.BytesIO(v))
      elif k == 'mp':
        ddata[k] = msgpack.unpackb(v)
      else:
        ddata[k] = v

    return ddata

  def generate(self):
    index, stream = 0, None
    try:
      while index < len(self._files):
        alog.debug(f'Opening new stream: {self._files[index]}')
        stream = StreamFile(self._files[index], **self._kwargs)
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

        alog.debug(f'Closing stream: {self._files[index]}')
        stream.close()
        stream = None
        index += 1
    finally:
      if stream is not None:
        alog.debug(f'Closing stream: {self._files[index]}')
        stream.close()

  def __iter__(self):
    return iter(self.generate())


def expand_files(url):
  m = re.match(r'(.*)\{(\d+)\.\.(\d+)\}(.*)', url)
  if m:
    isize = len(m.group(2))
    start = int(m.group(2))
    end = int(m.group(3))
    files = []
    for i in range(start, end + 1):
      files.append(m.group(1) + f'{i:0{isize}d}' + m.group(4))

    return files

  return [url]


def create(wds_train=None, wds_test=None, **kwargs):
  ds = dict()
  if wds_train:
    ds['train'] = WebDataset(**wds_train)
  if wds_test:
    ds['test'] = WebDataset(**wds_test)

  return ds

