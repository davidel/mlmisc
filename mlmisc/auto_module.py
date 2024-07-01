import functools

import torch
import torch.nn as nn


_CLASS = 'mclass'
_ARGS = 'args'
_KWARGS = 'kwargs'
_STATE = '__AM_ARGS__'
_MODARGS = '_create_args'


def _wrapped_state_dict(mod, *args, **kwargs):
  state = mod._saved_state_dict(*args, **kwargs)
  state[_STATE] = module_args(mod)

  return state


def _wrap_module(mod, create_args):
  setattr(mod, _MODARGS, create_args)

  mod._saved_state_dict = mod.state_dict
  mod.state_dict = functools.partial(_wrapped_state_dict, mod)

  return mod


def create(mclass, *args, **kwargs):
  mod = mclass(*args, **kwargs)

  create_args = {
    _CLASS: mclass,
    _ARGS: args,
    _KWARGS: kwargs,
  }

  return _wrap_module(mod, create_args)


def is_auto_state(state):
  return _STATE in state


def is_auto(mod):
  return hasattr(mod, _MODARGS)


def load(source, strict=True):
  state = source.copy() if isinstance(source, dict) else torch.load(source)

  create_args = state.pop(_STATE)

  lmod = create_args[_CLASS](*create_args[_ARGS], **create_args[_KWARGS])
  lmod.load_state_dict(state, strict=strict)
  lmod = _wrap_module(lmod, create_args)

  return lmod


def module_args(mod):
  return getattr(mod, _MODARGS)


def clone(mod):
  state = mod.state_dict()

  return load(state)


def new_as(mod):
  create_args = module_args(mod)

  lmod = create_args[_CLASS](*create_args[_ARGS], **create_args[_KWARGS])
  lmod = _wrap_module(lmod, create_args)

  return lmod

