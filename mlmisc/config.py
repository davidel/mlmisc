import re

import py_misc_utils.alog as alog
import py_misc_utils.module_utils as pymu
import py_misc_utils.utils as pyu


def _config_split(config_data):
  parts = pyu.resplit(config_data, ':')
  mod_config = pyu.parse_dict(parts[1], allow_args=True) if len(parts) == 2 else (dict(), ())

  return parts[0], mod_config


def _load_class(obj_path):
  m = re.match(r'(.*),([^\.]+)$', obj_path)
  if m:
    module = pymu.import_module(m.group(1))
    return pymu.module_getter(m.group(2))(module)

  return pymu.import_module_names(obj_path)


def create_object(name, config_data, *args, **kwargs):
  obj_path, (obj_config, obj_args) = _config_split(config_data)

  kwargs.update(obj_config)

  alog.debug(f'Creating {obj_path} {name} with: ({len(args)} API args) {obj_args} {kwargs}')

  obj_class = _load_class(obj_path)

  return obj_class(*(args + obj_args), **kwargs)


def create_optimizer(params, config_data, **kwargs):
  return create_object('optimizer', config_data, params, **kwargs)


def create_lr_scheduler(optimizer, config_data, **kwargs):
  return create_object('LR scheduler', config_data, optimizer, **kwargs)


def create_loss(config_data, **kwargs):
  return create_object('Loss', config_data, **kwargs)


def create_model(config_data, *args, **kwargs):
  return create_object('Model', config_data, *args, **kwargs)

