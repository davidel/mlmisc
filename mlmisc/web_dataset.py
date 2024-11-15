import collections
import io
import json
import os
import random
import re
import requests
import tarfile
import threading

import numpy as np
import py_misc_utils.alog as alog
import py_misc_utils.img_utils as pyimg
import py_misc_utils.utils as pyu
import torch

from . import dataset_base as dsb
from . import utils as ut


class StreamFile:

  def __init__(self, url, auth=None, chunk_size=1024 * 8, max_buffers=32, **kwargs):
    self._url = url
    self._auth = auth
    self._chunk_size = chunk_size
    self._max_buffers = max_buffers
    self._lock = threading.Lock()
    self._rcond = threading.Condition(self._lock)
    self._wcond = threading.Condition(self._lock)
    self._buffers = collections.deque()
    self._stopped = False

    headers = dict()
    if auth:
      headers['Authorization'] = auth

    self._response = requests.get(url, headers=headers, stream=True)
    self._response.raise_for_status()

    self._thread = threading.Thread(target=self._feed)
    self._thread.start()

  def _feed(self):
    for chunk in self._response.iter_content(chunk_size=self._chunk_size):
      with self._lock:
        self._buffers.append(chunk)
        self._rcond.notify()

        while len(self._buffers) > self._max_buffers and not self._stopped:
          self._wcond.wait()

        if self._stopped:
          break

    with self._lock:
      self._stopped = True
      self._rcond.notify()

  def stop(self):
    with self._lock:
      self._stopped = True
      self._rcond.notify()
      self._wcond.notify()

    self._thread.join()
    self._thread = None

  def read(self, size):
    iobuf = io.BytesIO()
    while size > 0:
      with self._lock:
        while not self._buffers and not self._stopped:
          self._rcond.wait()

        if self._buffers:
          buf = self._buffers[0]
          if size >= len(buf):
            iobuf.write(buf)

            self._buffers.popleft()
            self._wcond.notify()

            size -= len(buf)
          else:
            iobuf.write(buf[: size])
            self._buffers[0] = buf[size:]
            size = 0

        if self._stopped:
          break

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
    self._files = files

  def _decode(self, data):
    ddata = dict()
    for k, v in data.items():
      if k in {'jpg', 'png', 'jpeg'}:
        ddata[k] = pyimg.from_bytes(v)
      elif k == 'json':
        ddata[k] = json.loads(v)
      elif k == 'pth':
        ddata[k] = torch.load(io.BytesIO(v), weights_only=False)
      elif k in {'npy', 'npz'}:
        ddata[k] = np.load(io.BytesIO(v), allow_pickle=False)
      elif k in {'cls', 'cls2', 'index'}:
        ddata[k] = int(v)
      else:
        ddata[k] = v

    return ddata

  def generate(self):
    index = 0
    stream = None
    try:
      while index < len(self._files):
        alog.debug(f'Opening new stream: {self._files[index]}')
        stream = StreamFile(self._files[index], **self._kwargs)
        tar = tarfile.open(mode='r|', fileobj=stream)

        ctid, data = None, dict()
        for tinfo in tar:
          tid, text = os.path.splitext(tinfo.name)

          if tid != ctid and data:
            yield self._decode(data)
            data = dict()

          ctid = tid
          data[text[1:]] = tar.extractfile(tinfo).read()

        if data:
          yield self._decode(data)

        alog.debug(f'Closing stream: {self._files[index]}')
        stream.stop()
        stream = None
        index += 1
    finally:
      if stream is not None:
        alog.debug(f'Closing stream: {self._files[index]}')
        stream.stop()

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


def create(train=None, test=None):
  ds = dict()
  if train:
    ds['train'] = WebDataset(**train)
  if test:
    ds['test'] = WebDataset(**test)

  return ds

