import functools

import torch
import torch.nn as nn


_CLASS = 'mclass'
_ARGS = 'args'
_KWARGS = 'kwargs'
_STATE = '__AM_ARGS__'


def _wrapped_state_dict(mod, *args, **kwargs):
  state = mod._saved_state_dict(*args, **kwargs)
  state[_STATE] = mod._create_args

  return state


def _wrap_module(mod, create_args):
  mod._create_args = create_args

  mod._saved_state_dict = mod.state_dict
  mod.state_dict = functools.partial(_wrapped_state_dict, mod)

  return mod


def create_module(mclass, *args, **kwargs):
  mod = mclass(*args, **kwargs)

  create_args = {
    _CLASS: mclass,
    _ARGS: args,
    _KWARGS: kwargs,
  }

  return _wrap_module(mod, create_args)


def is_module(state):
  return _STATE in state


def load_module(state):
  if not isinstance(state, dict):
    state = torch.load(state)

  create_args = state.pop(_STATE)

  lmod = create_args[_CLASS](*create_args[_ARGS], **create_args[_KWARGS])
  lmod.load_state_dict(state)
  lmod = _wrap_module(lmod, create_args)

  return lmod


def module_args(mod):
  return mod._create_args

