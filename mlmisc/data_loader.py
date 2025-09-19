import functools
import multiprocessing
import pickle
import queue
import signal

import numpy as np
import py_misc_utils.alog as alog
import py_misc_utils.core_utils as pycu
import py_misc_utils.fin_wrap as pyfw
import py_misc_utils.multiprocessing as pymp
import py_misc_utils.num_utils as pynu
import py_misc_utils.pipeline as pypl
import py_misc_utils.signal as pysig
import py_misc_utils.utils as pyu
import torch
import torch.multiprocessing

from . import dataset_base as dsb
from . import dataset_utils as dsu


# We want to transfer exceptions from child processes, to parents, using the
# multiprocessing Quque objects, which require data to be pickle-able, and some
# of the generated exceptions contain data which is not.
class _QueueException(Exception):

  def __init__(self, ex):
    super().__init__(repr(ex))


class _QueueGetter:

  def __init__(self, input_queue, max_nones=1):
    self._input_queue = input_queue
    self._max_nones = max_nones
    self._nones = 0

  def get(self):
    while self._max_nones > self._nones:
      data = self._input_queue.get()
      if isinstance(data, Exception):
        raise data
      if data is not None:
        return data

      self._nones += 1


class _Expanded:

  def __init__(self, data):
    self.expanded = tuple(data)


class _PickledQueue:

  def __init__(self, mp_queue):
    self._queue = mp_queue

  def put(self, data):
    qdata = pickle.dumps(data)

    self._queue.put(qdata)

  def get(self, *args, **kwargs):
    qdata = self._queue.get(*args, **kwargs)

    return pickle.loads(qdata)

  def close(self):
    self._queue.close()

  def cancel_join_thread(self):
    pycu.maybe_call(self._queue, 'cancel_join_thread')


class _BatchCollater:

  def __init__(self, batch_size, collate_fn, indices):
    self._batch_size = batch_size
    self._collate_fn = collate_fn
    self._indices = np.asarray(indices)
    self._offset = 0
    self._batch = []
    self._cached = dict()

  def get_batch(self, force=False):
    next_offset = len(self._indices)
    for i in range(self._offset, len(self._indices)):
      index = self._indices[i]
      cdata = self._cached.pop(index, None)
      if cdata is not None:
        self._batch.extend(cdata)
      elif not force:
        next_offset = i
        break

    self._offset = next_offset
    if len(self._batch) >= self._batch_size or (force and self._batch):
      bdata = self._batch[: self._batch_size]
      self._batch = self._batch[self._batch_size:]

      return self._collate_fn(bdata), len(bdata)

  def add_indices(self, indices):
    self._indices = np.concatenate((self._indices[self._offset:], np.asarray(indices)))
    self._offset = 0

  def left_indices(self):
    return len(self._indices) - self._offset

  def add(self, batch):
    for index, data in batch:
      if isinstance(data, _Expanded):
        self._cached[index] = data.expanded
      else:
        self._cached[index] = (data,)

  def flush(self):
    return self.get_batch(force=True)


class _IterDataFeeder:

  def __init__(self, mpctx, dataset, input_queue, output_queues):
    self._dataset = dataset
    self._input_queue = input_queue
    self._output_queues = tuple(output_queues)
    self._proc = pymp.create_process(self._run, context=mpctx)
    self._proc.start()

  def _generate(self):
    data_sources = (self._dataset if isinstance(self._dataset, (list, tuple))
                    else (self._dataset,))
    for source in data_sources:
      yield from source

  def _run(self):
    _init_process()

    exit_result = None
    try:
      data_iter = self._generate()
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
      for output_queue in self._output_queues:
        output_queue.put(exit_result)
        output_queue.close()

  def close(self):
    self._input_queue.put(None)
    self._proc.join()


class _MapDataFeeder:

  def __init__(self, mpctx, dataset, input_queue, output_queue):
    self._dataset = dataset
    self._input_queue = input_queue
    self._output_queue = output_queue
    self._proc = pymp.create_process(self._run, context=mpctx)
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
      self._output_queue.close()

  def close(self):
    self._input_queue.put(None)
    self._proc.join()


class _DataTransformer:

  def __init__(self, mpctx, input_queue, output_queue, pipeline):
    self._input_queue = input_queue
    self._output_queue = output_queue
    self._pipeline = pipeline
    self._proc = pymp.create_process(self._run, context=mpctx)
    self._proc.start()

  def _process_data(self, index, data):
    # If the Pipeline returns an iterator, we need to unwrap it since it might
    # contain generator objects which are not pickle-able.
    xdata = _Expanded(data) if pycu.is_iterator(data) else data

    self._output_queue.put((index, xdata))

  def _run(self):
    _init_process()

    last_index, index_gap = 0, 0
    exit_result = None
    try:
      queue_getter = _QueueGetter(self._input_queue)

      while True:
        idata = queue_getter.get()
        if idata is None:
          break

        index, data = idata

        self._process_data(index, self._pipeline(data))

        index_gap, last_index = max(index_gap, index - last_index), index

      data = self._pipeline.flush()
      if data is not None:
        self._process_data(last_index + index_gap, data)

    except pypl.HaltedPipeline:
      pass
    except Exception as ex:
      alog.exception(ex, exmsg=f'Exception in data transformer')
      exit_result = _QueueException(ex)
    finally:
      self._output_queue.put(exit_result)
      self._output_queue.close()

  def close(self):
    self._input_queue.put(None)
    self._proc.join()


class _BareDataset(dsb.IterableDataset):

  def __init__(self, dataset):
    dsb.IterableDataset.__init__(self)
    self._datasets = pyu.as_sequence(dataset)
    self.add_sources(*self._datasets)

  def enum_samples(self):
    for dataset in self._datasets:
      if hasattr(dataset, 'enum_samples'):
        yield from dataset.enum_samples()
      else:
        yield from dataset


class _IterDataLoader:

  def __init__(self, mpctx, dataset, shuffle, batch_size, num_workers, drop_last,
               collate_fn, prefetch_factor, **kwargs):
    self._mpctx = mpctx
    self._dataset = dataset
    self._shuffle = shuffle
    self._batch_size = batch_size
    self._drop_last = drop_last
    self._collate_fn = collate_fn
    self._prefetch_factor = prefetch_factor
    self._input_queue = _create_queue(mpctx)
    self._output_queue = _create_queue(mpctx)
    self._trans_queues = []
    self._output_feeders = 0

    # In the case of an iterator dataset (that is, strictly sequential stream of
    # samples) we only have one _IterDataFeeder. If the dataset has a processing
    # pipeline it makes sense to have the _IterDataFeeder to feed N _DataTransformer
    # (whose task is the run the pipeline), which in turn feed the output queue.
    # Otherwise we have the _IterDataFeeder feed the output queue directly.
    # The _create_loader() API already trims the number of workers to one, in case
    # the input dataset is not an instance of dsb.IterableDataset (that is, it has
    # no pipeline).
    feeders = []
    if num_workers == 1:
      feeders.append(_IterDataFeeder(mpctx, dataset, self._input_queue,
                                     (self._output_queue,)))
      self._output_feeders += 1
    else:
      pipeline = dataset.pipeline().clone()

      transformers = []
      for i in range(num_workers - 1):
        self._trans_queues.append(_create_queue(mpctx))

        trs = _DataTransformer(mpctx, self._trans_queues[-1], self._output_queue,
                               pipeline)

        transformers.append(trs)
        self._output_feeders += 1

      pyfw.fin_wrap(self, '_transformers', transformers,
                    finfn=functools.partial(_closer, transformers))

      feeders.append(_IterDataFeeder(mpctx, _BareDataset(dataset), self._input_queue,
                                     self._trans_queues))

    pyfw.fin_wrap(self, '_feeders', feeders,
                  finfn=functools.partial(_closer, feeders))

  def close(self):
    pyfw.fin_wrap(self, '_feeders', None, cleanup=True)
    pyfw.fin_wrap(self, '_transformers', None, cleanup=True)

    for q in [self._input_queue, self._output_queue] + self._trans_queues:
      _queue_close(q)

  def _generate(self):
    idxgen = _IterIndexGenerator(self._batch_size, self._shuffle)
    queue_getter = _QueueGetter(self._output_queue, self._output_feeders)
    collater = _BatchCollater(self._batch_size, self._collate_fn, idxgen.generate())

    self._input_queue.put(self._prefetch_factor * self._batch_size)
    while True:
      self._input_queue.put(self._batch_size)

      if (indices := idxgen.generate(left=collater.left_indices())) is not None:
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

      if len(batch) < self._batch_size:
        break

    while (cbatch := collater.flush()) is not None:
      bdata, bsize = cbatch
      if bsize == self._batch_size or not self._drop_last:
        yield bdata

  def _iter_generate(self):
    try:
      yield from self._generate()
    finally:
      _queue_flush(self._output_queue)

  def __iter__(self):
    return self._iter_generate()

  def __len__(self):
    return _loader_size(self._dataset, self._batch_size, self._drop_last)


class _MapDataLoader:

  def __init__(self, mpctx, dataset, shuffle, batch_size, num_workers, drop_last,
               collate_fn, prefetch_factor, **kwargs):
    self._mpctx = mpctx
    self._dataset = dataset
    self._shuffle = shuffle
    self._batch_size = batch_size
    self._drop_last = drop_last
    self._collate_fn = collate_fn
    self._prefetch_factor = prefetch_factor
    self._input_queues = []
    self._output_queue = _create_queue(mpctx)

    feeders = []
    for i in range(num_workers):
      self._input_queues.append(_create_queue(mpctx))

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
    indices = np.arange(len(self._dataset))
    if self._shuffle:
      np.random.shuffle(indices)
    if self._drop_last:
      indices = indices[: pynu.round_down(len(indices), self._batch_size)]

    queue_getter = _QueueGetter(self._output_queue, len(self._input_queues))
    collater = _BatchCollater(self._batch_size, self._collate_fn, indices)

    index = self._feed_indices(indices, 0, self._prefetch_factor * self._batch_size)

    processed = 0
    while processed < len(indices):
      batch = []
      for i in range(min(self._batch_size, len(indices) - processed)):
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

      if len(batch) < self._batch_size:
        break

    while (cbatch := collater.flush()) is not None:
      bdata, bsize = cbatch
      yield bdata

  def _iter_generate(self):
    try:
      yield from self._generate()
    finally:
      _queue_flush(self._output_queue)

  def __iter__(self):
    return self._iter_generate()

  def __len__(self):
    return _loader_size(self._dataset, self._batch_size, self._drop_last)


class _SimpleDataLoader:

  def __init__(self, dataset, shuffle, batch_size, drop_last, collate_fn, **kwargs):
    self._dataset = dataset
    self._shuffle = shuffle
    self._batch_size = batch_size
    self._drop_last = drop_last
    self._collate_fn = collate_fn

  def close(self):
    pass

  def _map_generate(self):
    indices = np.arange(len(self._dataset))
    if self._shuffle:
      np.random.shuffle(indices)
    if self._drop_last:
      indices = indices[: pynu.round_down(len(indices), self._batch_size)]

    processed = 0
    while processed < len(indices):
      batch = []
      for i in range(processed, min(processed + self._batch_size, len(indices))):
        batch.append(self._dataset[indices[i]])

      processed += len(batch)

      yield self._collate_fn(batch)

  def _iter_generate(self):
    idxgen = _IterIndexGenerator(self._batch_size, self._shuffle)
    collater = _BatchCollater(self._batch_size, self._collate_fn, idxgen.generate())

    batch = []
    for index, data in enumerate(self._dataset):
      batch.append((index, data))

      if len(batch) == self._batch_size:
        collater.add(batch)
        batch = []

        while (cbatch := collater.get_batch()) is not None:
          bdata, bsize = cbatch
          yield bdata

        if (indices := idxgen.generate(left=collater.left_indices())) is not None:
          collater.add_indices(indices)

    if batch:
      collater.add(batch)

    while (cbatch := collater.flush()) is not None:
      bdata, bsize = cbatch
      if bsize == self._batch_size or not self._drop_last:
        yield bdata

  def __iter__(self):
    if isinstance(self._dataset, torch.utils.data.IterableDataset):
      return self._iter_generate()
    else:
      return self._map_generate()

  def __len__(self):
    return _loader_size(self._dataset, self._batch_size, self._drop_last)


class _IterIndexGenerator:

  REFILL_FACTOR = 4

  def __init__(self, batch_size, shuffle):
    self._shuffle = shuffle
    self._shuffle_window = batch_size * pyu.getenv('SHUFFLE_FACTOR', dtype=int, defval=16)
    self._size = 16 * self.REFILL_FACTOR * max(self._shuffle_window, 1)
    self._index = 0

  def generate(self, left=0):
    if self._size // self.REFILL_FACTOR >= left:
      csize = self._size - left
      indices = np.arange(self._index, self._index + csize)
      if self._shuffle and self._shuffle_window > 1:
        for idx in range(0, len(indices), self._shuffle_window):
          np.random.shuffle(indices[idx: idx + self._shuffle_window])

      self._index += csize

      return indices


# PyTorch hooks into the Python multiprocessing Queue implementation, by providing
# its own pickler, which uses shared memory (/dev/shm on Linux). This might be
# good when dealing with large dataset samples (ie, images), but on smaller ones
# (ie, tokens) not only it's slower but it tends to keep open too many files (or
# create too many mmap regions, when using the "file_system" sharing mode) which
# in turn result in OS errors.
# The _PickledQueue class is a simple wrapper around a multiprocessing Queue that
# tricks the PyTorch tensors multiprocessing serialization into emitting into a
# normal buffer, and exchange that within the Queue connection.
def _create_queue(mpctx):
  mp_queue = mpctx.Queue()

  use_torch_pickler = pyu.getenv('USE_TORCH_PICKLER', dtype=bool, defval=False)

  return mp_queue if use_torch_pickler else _PickledQueue(mp_queue)


def _queue_flush(mp_queue, timeout=1.0):
  try:
    while True:
      mp_queue.get(True, timeout)
  except queue.Empty:
    pass


def _queue_close(mp_queue):
  # Within child processes, all the queues used as output will have a new thread
  # created, which is used to flush the output data into the underline connection.
  # When such child process exists, by default it tries to join such thread, but
  # if the other side of the connection (queue transport) has quit fetching data,
  # it might hang while flushing the queue.
  #
  #   https://github.com/python/cpython/blob/8a7146c5eb340aa5115a5baf61e4f74c589d440f/Lib/multiprocessing/queues.py#L200
  #   https://github.com/python/cpython/blob/8a7146c5eb340aa5115a5baf61e4f74c589d440f/Lib/multiprocessing/queues.py#L213
  #
  # By using cancel_join_thread() will prevent the existing child process in trying
  # to join the flushing thread, and hence prevent possible hangs.
  pycu.maybe_call(mp_queue, 'cancel_join_thread')
  mp_queue.close()


def _closer(objs):
  for obj in objs:
    obj.close()


def _int_handler(sig, frame):
  return pysig.HANDLED


def _init_process():
  torch.set_num_threads(1)
  pysig.signal(signal.SIGINT, _int_handler)


def _loader_size(dataset, batch_size, drop_last):
  if (ds_size := dsu.dataset_size(dataset)) is not None:
    rounder = 0 if drop_last else batch_size - 1

    return (ds_size + rounder) // batch_size


def _get_batch_args(collate_fn, batch_size):
  if collate_fn is None:
    if batch_size == 0:
      collate_fn = pycu.ident
      batch_size = 1
    else:
      collate_fn = torch.utils.data.default_collate

  return batch_size, collate_fn


def _create_loader(mpctx, dataset, shuffle, batch_size, num_workers, drop_last,
                   collate_fn, prefetch_factor, **kwargs):
  batch_size, collate_fn = _get_batch_args(collate_fn, batch_size)

  if num_workers == 0:
    return _SimpleDataLoader(dataset, shuffle, batch_size, drop_last,
                             collate_fn, **kwargs)
  elif isinstance(dataset, torch.utils.data.IterableDataset):
    # If we have a number of workers greater than one, and the input dataset is
    # not a dsb.DatasetBase (or its pipeline is empty), force the number of workers
    # to be one since there is no need for extra.
    # This is because the architecture for a data loader fed with an iterable dataset,
    # is to have a single _IterDataFeeder (whose purpose is to simply read data and
    # write that to pipes) feeding multiple _DataTransformer instances, which read
    # from the pipes written by the _IterDataFeeder and run the pipeline on it.
    # If there is no pipeline, it makes no sense to pay for the extra _DataTransformer
    # layer.
    if num_workers > 1 and (not isinstance(dataset, dsb.DatasetBase) or
                            not dataset.pipeline()):
      alog.debug(f'Reducing the number of workers from {num_workers} to 1')
      num_workers = 1

    return _IterDataLoader(mpctx, dataset, shuffle, batch_size, num_workers, drop_last,
                           collate_fn, prefetch_factor, **kwargs)
  else:
    return _MapDataLoader(mpctx, dataset, shuffle, batch_size, num_workers, drop_last,
                          collate_fn, prefetch_factor, **kwargs)


# Why not using the PyTorch one?
# I have had many issues with different libraries liking or not liking certain
# Python multiprocessing start methods, for which the PyTorch DataLoader does not
# seem to play nice.
class DataLoader:

  def __init__(self, dataset,
               shuffle=False,
               batch_size=1,
               num_workers=1,
               drop_last=True,
               collate_fn=None,
               prefetch_factor=3,
               mpctx=torch.multiprocessing,
               **kwargs):
    loader = _create_loader(mpctx, dataset, shuffle, batch_size, num_workers,
                            drop_last, collate_fn, prefetch_factor, **kwargs)
    pyfw.fin_wrap(self, '_loader', loader, finfn=loader.close)

  def close(self):
    pyfw.fin_wrap(self, '_loader', None, cleanup=True)

  def _generate(self):
    alog.debug(f'DataLoader generator started')
    try:
      # Avoid using `yield from self._loader` since this will not keep a DataLoader
      # reference around, so the _loader could be closed while there is a Generator
      # alive.
      for data in self._loader:
        yield data
    finally:
      alog.debug(f'DataLoader generator exit')

  def __iter__(self):
    return self._generate()

  def __len__(self):
    return len(self._loader)

