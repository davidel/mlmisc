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

def _compile(sfn, info, env):
  if sfn.find('\n') < 0:
    sfn = f'lambda {sfn}'
    fn = eval(sfn, env)
  else:
    sfn = sfn.strip()
    fn, = pyu.compile(sfn, NETFN, env=env)

  return fn, info or sfn


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

