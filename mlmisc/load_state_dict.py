import py_misc_utils.alog as alog
import torch


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
    result = module.load_state_dict(state, strict=strict, **kwargs)
  elif strict == 'force':
    result = None
    try:
      result = module.load_state_dict(state, strict=False, **kwargs)
    except Exception as ex:
      alog.warning(f'{ex}')
  else:
    alog.xraise(ValueError, f'Wrong argument value for "strict": {strict}')

  return result
