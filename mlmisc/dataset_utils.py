import py_misc_utils.alog as alog
import torch

from . import utils as ut


def get_class_weights(dataset, dtype=None, cdtype=None, output_filter=None):
  output_filter = output_filter or (lambda x: x[1])
  target = torch.empty(len(dataset), dtype=cdtype or torch.int32)
  for i in range(len(dataset)):
    y = output_filter(dataset[i])
    target[i] = ut.item(y)

  cvalues, class_counts = torch.unique(target, return_counts=True)
  weight = 1.0 / class_counts
  weight = weight / torch.sum(weight)

  if dtype is not None:
    weight = weight.to(dtype)

  if ut.is_integer(cvalues):
    max_class = torch.max(cvalues).item()
    if max_class >= len(cvalues):
      fweight = torch.zeros(max_class + 1, dtype=weight.dtype)
      fweight[cvalues] = weight
      weight = fweight

  alog.debug(f'Dataset class weight: { {c: f"{n:.2e}" for c, n in enumerate(weight)} }')

  return weight

