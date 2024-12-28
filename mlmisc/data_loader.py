import functools
import multiprocessing
import queue

import numpy as np
import py_misc_utils.alog as alog
import py_misc_utils.app_main as app_main
import py_misc_utils.core_utils as pycu
import py_misc_utils.fin_wrap as pyfw
import py_misc_utils.utils as pyu
import torch

from . import dataset_base as dsb
from . import dataset_utils as dsu


class _QueueException(Exception):

  def __init__(self, ex):
    super().__init__(repr(ex))


class _QueueGetter:

  def __init__(self, input_queue, max_nones=1):
    self._input_queue = input_queue
    self._max_nones = max_nones
    self._nones = 0

  def get(self):
    data = None
    while self._max_nones > self._nones:
      data = self._input_queue.get()
      if isinstance(data, Exception):
        raise data
      if data is not None:
        break

      self._nones += 1

    return data


class _BatchCollater:

  def __init__(self, batch_size, collate_fn, indices):
    self._batch_size = batch_size
    self._collate_fn = collate_fn
    self._indices = np.asarray(indices)
    self._index = 0
    self._pending = set(self._indices[: batch_size])
    self._cached = dict()

  def _make_batch(self):
    bdata = []
    for index in self._indices[self._index: self._index + self._batch_size]:
      data = self._cached.pop(index, None)
      if data is not None:
        bdata.append(data)
        if len(bdata) == self._batch_size:
          break

    if bdata:
      self._index += self._batch_size
      self._pending = set(self._indices[self._index: self._index + self._batch_size]) - \
        set(self._cached.keys())

      return self._collate_fn(bdata), len(bdata)

  def add_indices(self, indices):
    self._indices = np.concatenate((self._indices[self._index:], np.asarray(indices)))
    self._index = 0

    return len(self._indices)

  def left_indices(self):
    return len(self._indices) - self._index

  def add(self, batch):
    for index, data in batch:
      self._cached[index] = data
      self._pending.discard(index)

  def get_batch(self):
    return None if self._pending else self._make_batch()

  def flush(self):
    return self._make_batch()


class _IterDataFeeder:

  def __init__(self, mpctx, dataset, input_queue, output_queues):
    self._dataset = dataset
    self._input_queue = input_queue
    self._output_queues = output_queues
    self._proc = mpctx.Process(target=app_main.wrap_main(self._run))
    self._proc.start()

  def _generate(self):
    if isinstance(self._dataset, dsb.IterableDataset):
      yield from self._dataset.enum_samples()
    else:
      yield from self._dataset

  def _run(self):
    _init_process()

    exit_result = None
    try:
      data_iter = iter(self._generate())

      queue_getter = _QueueGetter(self._input_queue)

      index = 0
      while True:
        feed_size = queue_getter.get()
        if feed_size is None:
          break

        for i in range(feed_size):
          data = next(data_iter)

          output_queue = self._output_queues[index % len(self._output_queues)]

          output_queue.put((index, data))
          index += 1
    except StopIteration:
      pass
    except Exception as ex:
      alog.exception(ex, exmsg=f'Exception in data loader iter data feeder')
      exit_result = _QueueException(ex)
    finally:
      for outq in self._output_queues:
        outq.put(exit_result)

  def close(self):
    self._input_queue.put(None)
    self._proc.join()


class _MapDataFeeder:

  def __init__(self, mpctx, dataset, input_queue, output_queue):
    self._dataset = dataset
    self._input_queue = input_queue
    self._output_queue = output_queue
    self._proc = mpctx.Process(target=app_main.wrap_main(self._run))
    self._proc.start()

  def _run(self):
    _init_process()

    exit_result = None
    try:
      queue_getter = _QueueGetter(self._input_queue)

      while True:
        indices = queue_getter.get()
        if indices is None:
          break

        for index in indices:
          data = self._dataset[index]

          self._output_queue.put((index, data))
    except Exception as ex:
      alog.exception(ex, exmsg=f'Exception in data loader map data feeder')
      exit_result = _QueueException(ex)
    finally:
      self._output_queue.put(exit_result)

  def close(self):
    self._input_queue.put(None)
    self._proc.join()


class _DataTransformer:

  def __init__(self, mpctx, input_queue, output_queue, pipeline):
    self._input_queue = input_queue
    self._output_queue = output_queue
    self._pipeline = pipeline
    self._proc = mpctx.Process(target=app_main.wrap_main(self._run))
    self._proc.start()

  def _run(self):
    _init_process()

    exit_result = None
    try:
      queue_getter = _QueueGetter(self._input_queue)

      while True:
        idata = queue_getter.get()
        if idata is None:
          break

        index, data = idata

        data = self._pipeline(data)

        self._output_queue.put((index, data))
    except Exception as ex:
      alog.exception(ex, exmsg=f'Exception in data transformer')
      exit_result = _QueueException(ex)
    finally:
      self._output_queue.put(exit_result)

  def close(self):
    self._input_queue.put(None)
    self._proc.join()


class _IterDataLoader:

  def __init__(self, mpctx, dataset, shuffle, batch_size, num_workers, collate_fn,
               prefetch_factor, shuffle_window=None, **kwargs):
    self._mpctx = mpctx
    self._dataset = dataset
    self._shuffle = shuffle
    self._batch_size = batch_size
    self._collate_fn = collate_fn
    self._prefetch_factor = prefetch_factor
    self._shuffle_window = pyu.value_or(shuffle_window, 512)
    self._input_queue = mpctx.Queue()
    self._output_queue = mpctx.Queue()
    self._trans_queues = []

    if num_workers == 1:
      feeder = _IterDataFeeder(mpctx, dataset, self._input_queue, (self._output_queue,))
      pyfw.fin_wrap(self, '_feeder', feeder, finfn=feeder.close)
    else:
      pipeline = dataset.pipeline()
      transformers = []
      for i in range(num_workers - 1):
        self._trans_queues.append(mpctx.Queue())

        trs = _DataTransformer(mpctx, self._trans_queues[-1], self._output_queue,
                               pipeline)

        transformers.append(trs)

      pyfw.fin_wrap(self, '_transformers', transformers,
                    finfn=functools.partial(_closer, transformers))

      feeder = _IterDataFeeder(mpctx, dataset, self._input_queue, self._trans_queues)
      pyfw.fin_wrap(self, '_feeder', feeder, finfn=feeder.close)

  def close(self):
    pyfw.fin_wrap(self, '_feeder', None, cleanup=True)
    pyfw.fin_wrap(self, '_transformers', None, cleanup=True)

    for q in [self._input_queue, self._output_queue] + self._trans_queues:
      _queue_close(q)

  def _make_indices(self, index, size):
    indices = np.arange(index, index + size)
    if self._shuffle:
      for idx in range(0, len(indices), self._shuffle_window):
        np.random.shuffle(indices[idx: idx + self._shuffle_window])

    return indices, index + size

  def _generate(self):
    indices_size = max(10000, 8 * self._shuffle_window)
    try:
      indices, index = self._make_indices(0, indices_size)

      queue_getter = _QueueGetter(self._output_queue, max(1, len(self._trans_queues)))

      collater = _BatchCollater(self._batch_size, self._collate_fn, indices)

      self._input_queue.put(self._prefetch_factor * self._batch_size)
      while True:
        self._input_queue.put(self._batch_size)

        if indices_size // 3 > collater.left_indices():
          indices, index = self._make_indices(index, indices_size - collater.left_indices())
          collater.add_indices(indices)

        batch = []
        for i in range(self._batch_size):
          idata = queue_getter.get()
          if idata is None:
            break

          batch.append(idata)

        if batch:
          collater.add(batch)

          while (cbatch := collater.get_batch()) is not None:
            bdata, bsize = cbatch
            yield bdata

            del cbatch

        if len(batch) < self._batch_size:
          break

      while (cbatch := collater.flush()) is not None:
        bdata, bsize = cbatch
        yield bdata
    except StopIteration:
      pass
    finally:
      pass

  def __iter__(self):
    return iter(self._generate())

  def __len__(self):
    return dsu.dataset_size(self._dataset)


class _MapDataLoader:

  def __init__(self, mpctx, dataset, shuffle, batch_size, num_workers, collate_fn,
               prefetch_factor, **kwargs):
    self._mpctx = mpctx
    self._dataset = dataset
    self._shuffle = shuffle
    self._batch_size = batch_size
    self._collate_fn = collate_fn
    self._prefetch_factor = prefetch_factor
    self._input_queues = []
    self._output_queue = mpctx.Queue()

    feeders = []
    for i in range(num_workers):
      self._input_queues.append(mpctx.Queue())

      feeder = _MapDataFeeder(mpctx, dataset, self._input_queues[-1], self._output_queue)

      feeders.append(feeder)

    pyfw.fin_wrap(self, '_feeders', feeders,
                  finfn=functools.partial(_closer, feeders))

  def close(self):
    pyfw.fin_wrap(self, '_feeders', None, cleanup=True)

    for q in self._input_queues + [self._output_queue]:
      _queue_close(q)

  def _feed_indices(self, indices, index, n):
    stop = min(index + n, len(indices))
    for i in range(index, stop):
      input_queue = self._input_queues[i % len(self._input_queues)]

      input_queue.put([indices[i]])

    return stop

  def _generate(self):
    try:
      indices = np.arange(len(self._dataset))
      if self._shuffle:
        np.random.shuffle(indices)

      queue_getter = _QueueGetter(self._output_queue, len(self._input_queues))

      collater = _BatchCollater(self._batch_size, self._collate_fn, indices)

      if self._prefetch_factor:
        index = self._feed_indices(indices, 0, self._prefetch_factor * self._batch_size)
      else:
        index = 0

      processed = 0
      while processed < len(indices):
        batch = []
        for i in range(self._batch_size):
          idata = queue_getter.get()
          if idata is None:
            break

          batch.append(idata)

        if batch:
          collater.add(batch)

          while (cbatch := collater.get_batch()) is not None:
            bdata, bsize = cbatch
            processed += bsize
            index = self._feed_indices(indices, index, self._batch_size)

            yield bdata

            del cbatch

        if len(batch) < self._batch_size:
          break

      while (cbatch := collater.flush()) is not None:
        bdata, bsize = cbatch
        yield bdata
    except StopIteration:
      pass
    finally:
      pass

  def __iter__(self):
    return iter(self._generate())

  def __len__(self):
    return dsu.dataset_size(self._dataset)


def _queue_close(q):
  q.cancel_join_thread()
  q.close()


def _closer(objs):
  for obj in objs:
    obj.close()


def _init_process():
  torch.set_num_threads(1)


def _create_loader(mpctx, dataset, shuffle, batch_size, num_workers, collate_fn,
                   prefetch_factor, **kwargs):
  if isinstance(dataset, torch.utils.data.IterableDataset):
    if num_workers > 1 and not isinstance(dataset, dsb.DatasetBase):
      num_workers = 1

    return _IterDataLoader(mpctx, dataset, shuffle, batch_size, num_workers, collate_fn,
                           prefetch_factor, **kwargs)
  else:
    return _MapDataLoader(mpctx, dataset, shuffle, batch_size, num_workers, collate_fn,
                          prefetch_factor, **kwargs)


class DataLoader:

  def __init__(self, dataset,
               shuffle=None,
               batch_size=None,
               num_workers=None,
               collate_fn=None,
               prefetch_factor=None,
               mpctx=None,
               **kwargs):
    shuffle = pyu.value_or(shuffle, False)
    batch_size = pyu.value_or(batch_size, 16)
    num_workers = pyu.value_or(num_workers, 1)
    collate_fn = pyu.value_or(collate_fn, torch.utils.data.default_collate)
    prefetch_factor = max(pyu.value_or(prefetch_factor, 3), 1)
    mpctx = pyu.value_or(mpctx, multiprocessing)

    loader = _create_loader(mpctx, dataset, shuffle, batch_size, num_workers, collate_fn,
                            prefetch_factor, **kwargs)
    pyfw.fin_wrap(self, '_loader', loader, finfn=loader.close)

  def close(self):
    pyfw.fin_wrap(self, '_loader', None, cleanup=True)

  def __iter__(self):
    return iter(self._loader)

  def __len__(self):
    return len(self._loader)

