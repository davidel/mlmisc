import functools

import torch
import torch.nn as nn


_CLASS_KEY = 'mclass'
_ARGS_KEY = 'args'
_KWARGS_KEY = 'kwargs'
_STATE_KEY = 'AUTO_MODULE_ARGS'


def _wrapped_state_dict(mod, *args, **kwargs):
  state = mod._saved_state_dict(*args, **kwargs)
  state[_STATE_KEY] = mod._create_args

  return state


def _wrap_module(mod, create_args):
  mod._create_args = create_args

  mod._saved_state_dict = mod.state_dict
  mod.state_dict = functools.partial(_wrapped_state_dict, mod)

  return mod


def create_module(mclass, *args, **kwargs):
  mod = mclass(*args, **kwargs)

  return _wrap_module(mod, dict(mclass=mclass, args=args, kwargs=kwargs))


def is_module(state):
  return _STATE_KEY in state


def load_module(state):
  create_args = state.pop(_STATE_KEY)

  lmod = create_args[_CLASS_KEY](*create_args[_ARGS_KEY], **create_args[_KWARGS_KEY])
  lmod.load_state_dict(state)
  lmod = _wrap_module(lmod, create_args)

  return lmod


def module_args(mod):
  return mod._create_args

