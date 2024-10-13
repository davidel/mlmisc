import functools
import io
import pickle

import py_misc_utils.module_utils as pymu
import py_misc_utils.utils as pyu
import torch
import torch.nn as nn


_CLASS = 'mclass'
_ARGS = 'args'
_KWARGS = 'kwargs'
_STATE = '__AM_ARGS__'
_MODARGS = '_create_args'


def _save_state(state):
  bio = io.BytesIO()
  pickle.dump(state, bio)

  return bio.getvalue()


def _load_state(data):
  if isinstance(data, dict):
    return data

  bio = io.BytesIO(data)

  return pickle.load(bio)


def _wrapped_state_dict(mod, *args, **kwargs):
  state = mod._saved_state_dict(*args, **kwargs)
  state[_STATE] = _save_state(module_args(mod))

  return state


def _wrap_module(mod, create_args):
  setattr(mod, _MODARGS, create_args)

  mod._saved_state_dict = mod.state_dict
  mod.state_dict = functools.partial(_wrapped_state_dict, mod)

  return mod


def _generate_wrapped(create_args):
  ctor = create_args[_CLASS]
  # TODO: Remove dual handling till we have other-than-strings ctors.
  if isinstance(ctor, str):
    ctor, = pymu.import_module_names(ctor)

  mod = ctor(*create_args[_ARGS], **create_args[_KWARGS])

  return _wrap_module(mod, create_args)


def create(mclass, *args, **kwargs):
  create_args = {
    _CLASS: pyu.qual_name(mclass),
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

  create_args = _load_state(state.pop(_STATE))

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

