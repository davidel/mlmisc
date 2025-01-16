import py_misc_utils.assert_checks as tas
import py_misc_utils.core_utils as pycu
import py_misc_utils.obj as obj
import torch
import torch.nn as nn

from . import args_sequential as aseq


class ResultsNamespace(aseq.ArgsSequential):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._nss = []

  def ns_new(self):
    self._nss.append(obj.Obj())

    return self._nss[-1]

  def ns_len(self):
    return len(self._nss)

  def ns_get(self, i):
    return self._nss[i]

  def ns_clear(self):
    for ns in self._nss:
      vars(ns).clear()

  def forward(self, *args, **kwargs):
    y = super().forward(*args, **kwargs)

    self.ns_clear()

    return y

