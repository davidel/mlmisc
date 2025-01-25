import numpy as np
import pandas as pd
import py_misc_utils.alog as alog
import py_misc_utils.core_utils as pycu
import py_misc_utils.np_utils as pyn
import py_misc_utils.obj as pyobj


class Memory:

  def __init__(self, fields, capacity, dtype=np.float32):
    self._buffers = dict()
    for name, size in fields.items():
      if isinstance(size, int):
        bshape = (size,)
      elif isinstance(ss, list):
        bshape = tuple(size)

      if isinstance(dtype, dict):
        bdtype = dtype.get(name)
      else:
        bdtype = dtype

      self._buffers[name] = pyn.RingBuffer(capacity, bdtype, bshape)

  def _make_record(self, idx, trans=pycu.ident):
    rec = pyobj.Obj()
    for name, buffer in self._buffers.items():
      setattr(rec, name, trans(buffer[idx]))

    return rec

  def __len__(self):
    return min(len(buffer) for buffer in self._buffers.values())

  def __getitem__(self, i):
    return self._make_record(i)

  def capacity(self):
    return min(buffer.capacity for buffer in self._buffers.values())

  def resize(self, capacity):
    for buffer in self._buffers.values():
      buffer.resize(capacity)

  def append(self, **kwargs):
    for name, value in kwargs.items():
      self._buffers[name].append(value)

  def gen_samples(self, batch_size, trans=pycu.ident):
    indices = np.arange(len(self))
    np.random.shuffle(indices)
    for batch_indices in np.array_split(indices, batch_size):
      yield self._make_record(batch_indices, trans=trans)

  def iter_samples(self, *args, **kwargs):
    return iter(self.gen_samples(*args, **kwargs))

  def dataframe(self, expanded=True):
    data = dict()
    for name, buffer in self._buffers.items():
      arr = buffer.to_numpy().astype(np.float64)
      if arr.shape[-1] == 1:
        data[name] = arr.squeeze(-1)
      else:
        for i in range(arr.shape[-1]):
          data[f'{name}.{i}'] = arr[..., i]

    return pd.DataFrame(data=data)

  def filter(self, fields, factor):
    sums = 0
    for field in fields:
      data = self._buffers[field].data(dtype=np.float32)
      fdevs = pyn.normalize(data, axis=0)
      sums = np.sum(fdevs**2, axis=1) + sums

    probs = sums / sums.sum()

    new_size = int(len(sums) * factor)
    indices = np.random.choice(np.arange(len(sums)),
                               size=new_size,
                               replace=False,
                               p=probs)

    for name, buffer in self._buffers.items():
      buffer.select(indices)

