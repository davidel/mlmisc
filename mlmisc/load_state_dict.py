import collections

import py_misc_utils.alog as alog
import torch


LoadResult = collections.namedtuple(
  'LoadResult',
  'missing_keys, unexpected_keys, exception',
  defaults=(None,),
)

VALID_STRICTS = {
  'true': True,
  'false': False,
  '1': True,
  '0': False,
  'force': 'force',
}

def load_state_dict(module, state, strict=None, **kwargs):
  if strict is None:
    strict = True
  elif isinstance(strict, str):
    strict = VALID_STRICTS[strict.lower()]

  if isinstance(strict, bool):
    lsd_res = module.load_state_dict(state, strict=strict, **kwargs)
    result = LoadResult(missing_keys=lsd_res.missing_keys,
                        unexpected_keys=lsd_res.unexpected_keys)
  elif strict == 'force':
    result = None
    try:
      lsd_res = module.load_state_dict(state, strict=False, **kwargs)
      result = LoadResult(missing_keys=lsd_res.missing_keys,
                          unexpected_keys=lsd_res.unexpected_keys)
    except Exception as ex:
      alog.warning(f'{ex}')
      result = LoadResult(missing_keys=(),
                          unexpected_keys=(),
                          exception=ex)
  else:
    alog.xraise(ValueError, f'Wrong argument value for "strict": {strict}')

  for pk in result.missing_keys:
    alog.debug(f'Missing parameter: {pk}')
  for pk in result.unexpected_keys:
    alog.debug(f'Extra parameter: {pk}')

  return result


def save_obj_state(obj, state, state_fields=None, no_state_fields=None, key=None):
  state_fields = set(pyu.expand_strings(state_fields or ()))
  no_state_fields = set(pyu.expand_strings(no_state_fields or ()))
  key = pyu.value_or(key, pyu.cname(obj))

  obj_state = dict()
  for k, v in obj.__dict__.items():
    if ((state_fields and k in state_fields) or
        (no_state_fields and k not in no_state_fields)):
      obj_state[k] = v

  state[key] = obj_state


def load_obj_state(obj, state, key=None):
  key = pyu.value_or(key, pyu.cname(obj))
  obj.__dict__.update(state.pop(key, dict()))

