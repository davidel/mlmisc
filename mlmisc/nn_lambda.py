import inspect

import py_misc_utils.inspect_utils as pyiu
import py_misc_utils.utils as pyu
import torch
import torch.nn as nn


def _try_getsource(fn):
  # Do not fail if inspect is not able to fetch the source.
  try:
    return inspect.getsource(fn)
  except:
    pass


NETFN = 'netfn'

def _compile(fn, info, env):
  if fn.find('\n') < 0:
    return eval(f'lambda {fn}', env), info or f'lambda {fn}'
  else:
    sfn = fn.strip()
    return pyu.compile(sfn, NETFN)[0], info or sfn


class Lambda(nn.Module):

  def __init__(self, fn, info=None, env=None):
    super().__init__()
    if isinstance(fn, str):
      genv = env if env is not None else pyiu.parent_globals()
      self.fn, self.info = _compile(fn, info, genv)
    else:
      self.fn = fn
      self.info = info if info is not None else _try_getsource(fn)

  def forward(self, *args, **kwargs):
    return self.fn(*args, **kwargs)

  def extra_repr(self):
    return self.info or ''

