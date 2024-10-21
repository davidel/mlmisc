import functools
import io
import pickle

import torch

from . import load_state_dict as lsd


_CLASS = 'mclass'
_ARGS = 'args'
_KWARGS = 'kwargs'
_SAVED_STATE_DICT = '_saved_state_dict'
_STATE = '__AM_ARGS__'
_MODULE_ARGS = '_create_args'


def _save_state(state):
  bio = io.BytesIO()
  pickle.dump(state, bio)

  return bio.getvalue()


def _load_state(data):
  if isinstance(data, dict):
    return data

  bio = io.BytesIO(data)

  return pickle.load(bio)


def raw_state_dict(module, *args, **kwargs):
  state_dict = getattr(module, _SAVED_STATE_DICT, None)
  if state_dict is None:
    return module.state_dict(*args, **kwargs)

  return state_dict(*args, **kwargs)


def _wrapped_state_dict(module, *args, **kwargs):
  state = raw_state_dict(module, *args, **kwargs)
  state[_STATE] = _save_state(module_args(module))

  return state


def _wrap_module(module, create_args):
  setattr(module, _MODULE_ARGS, create_args)
  setattr(module, _SAVED_STATE_DICT, module.state_dict)
  module.state_dict = functools.partial(_wrapped_state_dict, module)

  return module


def _generate_wrapped(create_args):
  module = create_args[_CLASS](*create_args[_ARGS], **create_args[_KWARGS])

  return _wrap_module(module, create_args)


def create(mclass, *args, **kwargs):
  create_args = {
    _CLASS: mclass,
    _ARGS: args,
    _KWARGS: kwargs,
  }

  return _generate_wrapped(create_args)


def is_auto_state(state):
  return _STATE in state


def purged_state(state):
  state.pop(_STATE, None)

  return state


def is_auto(module):
  return hasattr(module, _MODULE_ARGS)


def load(source, map_location=None, strict=None):
  if isinstance(source, dict):
    state = source.copy()
  else:
    state = torch.load(source, map_location=map_location, weights_only=False)

  create_args = _load_state(state.pop(_STATE))

  module = _generate_wrapped(create_args)
  lsd.load_state_dict(module, state, strict=strict)

  return module


def module_args(module):
  return getattr(module, _MODULE_ARGS)


def clone(module):
  state = module.state_dict()

  return load(state)


def new_as(module):
  create_args = module_args(module)

  return _generate_wrapped(create_args)

