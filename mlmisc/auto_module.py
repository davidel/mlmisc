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


def _generate_wrapped(create_args):
  mod = create_args[_CLASS](*create_args[_ARGS], **create_args[_KWARGS])

  return _wrap_module(mod, create_args)


def create(mclass, *args, **kwargs):
  create_args = {
    _CLASS: mclass,
    _ARGS: args,
    _KWARGS: kwargs,
  }

  return _generate_wrapped(create_args)


def is_auto_state(state):
  return _STATE in state


def is_auto(mod):
  return hasattr(mod, _MODARGS)


def load(source, map_location=None, strict=True):
  if isinstance(source, dict):
    state = source.copy()
  else:
    state = torch.load(source, map_location=map_location, weights_only=False)

  create_args = state.pop(_STATE)

  mod = _generate_wrapped(create_args)
  mod.load_state_dict(state, strict=strict)

  return mod


def module_args(mod):
  return getattr(mod, _MODARGS)


def clone(mod):
  state = mod.state_dict()

  return load(state)


def new_as(mod):
  create_args = module_args(mod)

  return _generate_wrapped(create_args)

