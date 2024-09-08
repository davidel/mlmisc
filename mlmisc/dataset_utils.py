import numpy as np
import py_misc_utils.alog as alog
import torch


def get_class_weights(dataset):
  target = np.empty(dtype=np.int32)
  for i in range(len(dataset)):
    x, y = dataset[i]
    target[i] mlut.item(y)

  class_counts = np.unique(target, return_counts=True)[1]
  weight = class_counts / np.sum(class_counts)

  ds_weight = torch.tensor(weight, dtype=torch.float)

  alog.debug(f'Dataset class weight: { {c: f"{n * 100:.2f}%" for c, n in enumerate(ds_weight)} }')

  return ds_weight

