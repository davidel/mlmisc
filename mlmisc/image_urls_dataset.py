import py_misc_utils.alog as alog
import py_misc_utils.compression as pycomp
import py_misc_utils.http_async_fetcher as pyhaf
import py_misc_utils.img_utils as pyimg
import py_misc_utils.utils as pyu
import py_misc_utils.work_results as pywres
import torch

from . import dataset_base as dsb


class ImageUrlsDataset(torch.utils.data.IterableDataset):

  def __init__(self, urls, **kwargs):
    super().__init__()
    self._urls = tuple(urls)
    self._kwargs = kwargs

  def generate(self):
    num_workers = self._kwargs.get('num_workers')
    http_args = self._kwargs.get('http_args')
    if http_args is None:
      http_args = pyu.dict_subset(self._kwargs, 'headers', 'timeout')

    queue_batch = self._kwargs.get('queue_batch', 256)
    with pyhaf.HttpAsyncFetcher(num_workers=num_workers, http_args=http_args) as haf:
      index = queued = 0
      while index < len(self._urls):
        qcap = min(queue_batch - queued, len(self._urls) - index)
        qlist = self._urls[index: index + qcap]

        haf.enqueue(*qlist)

        queued += len(qlist)
        index += len(qlist)

        for url, data in haf.iter_results(max_results=queued // 2):
          queued -= 1
          if not isinstance(data, pywres.WorkException):
            try:
              yield pyimg.from_bytes(data)
            except:
              pass

  def __iter__(self):
    return iter(self.generate())

  def __len__(self):
    return len(self._urls)


def create(urls_path,
           shuffle=None,
           split_pct=None,
           seed=None,
           **kwargs):
  split_pct = pyu.value_or(split_pct, 0.9)

  urls = set()
  with pycomp.dopen(urls_path, mode='rt', **kwargs) as fd:
    for url in fd:
      url = url.strip()
      if url:
        urls.add(url)

  urls = sorted(urls)
  if shuffle in (True, None):
    # Stable shuffling, given same seed. Even though the ImageUrlsDataset (and the
    # ShufflerDataset) do shuffle urls/samples, because of the way we split
    # between train/test urls (by slicing), randomization is needed since the
    # distribution might not be uniform among the dataset urls.
    urls = dsb.shuffled_data(urls, seed=seed)

  ntrain = int(split_pct * len(urls))
  train_urls = urls[: ntrain]
  test_urls = urls[ntrain:]

  ds = dict()
  ds['train'] = ImageUrlsDataset(train_urls, **kwargs)
  ds['test'] = ImageUrlsDataset(test_urls, **kwargs)

  return ds

