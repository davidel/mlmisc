import random

import py_misc_utils.alog as alog
import py_misc_utils.gfs as gfs
import py_misc_utils.img_utils as pyimg
import py_misc_utils.url_fetcher as pyuf
import py_misc_utils.utils as pyu
import py_misc_utils.work_results as pywres
import torch

from . import dataset_base as dsb
from . import dataset_utils as dsu


class ImageUrlsDataset(torch.utils.data.IterableDataset):

  def __init__(self, urls, shuffle=True, **kwargs):
    super().__init__()
    self._urls = tuple(urls)
    self._shuffle = shuffle
    self._kwargs = kwargs

  def generate(self):
    if self._shuffle:
      urls = random.sample(self._urls, len(self._urls))
    else:
      urls = self._urls

    convert = self._kwargs.get('convert')
    queue_batch = self._kwargs.get('queue_batch', 256)
    num_workers = self._kwargs.get('num_workers', queue_batch)

    with pyuf.UrlFetcher(num_workers=num_workers, fs_kwargs=self._kwargs) as urlf:
      index = queued = 0
      while index < len(urls):
        qcap = min(queue_batch - queued, len(urls) - index)
        qlist = urls[index: index + qcap]

        urlf.enqueue(*qlist)

        queued += len(qlist)
        index += len(qlist)

        for url, data in urlf.iter_results(max_results=queued // 2):
          queued -= 1

          exception = None
          if not isinstance(data, pywres.WorkException):
            try:
              img = pyimg.from_bytes(data, convert=convert)
              yield (img,)
            except GeneratorExit:
              raise
            except Exception as ex:
              exception = ex
          else:
            exception = data.exception()

          if exception is not None:
            pyu.mlog(lambda: f'Exception: {exception}', level=alog.VERBOSE)

  def __iter__(self):
    return self.generate()

  def __len__(self):
    return len(self._urls)


def create(urls_path,
           url_shuffle=True,
           shuffle=True,
           train_pct=0.9,
           seed=None,
           **kwargs):
  urls = dsu.expand_urls(urls_path)

  if url_shuffle:
    # Stable shuffling, given same seed. Even though the ImageUrlsDataset do shuffle
    # urls/samples, because of the way we split between train/test urls (by slicing),
    # randomization is needed since the distribution might not be uniform among the
    # dataset urls.
    urls = dsb.shuffled_data(urls, seed=seed)

  ntrain = int(train_pct * len(urls))
  train_urls = urls[: ntrain]
  test_urls = urls[ntrain:]

  ds = dict()
  ds['train'] = ImageUrlsDataset(train_urls, shuffle=shuffle, **kwargs)
  ds['test'] = ImageUrlsDataset(test_urls, shuffle=shuffle, **kwargs)

  return ds

