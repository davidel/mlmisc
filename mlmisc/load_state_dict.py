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

  return result

