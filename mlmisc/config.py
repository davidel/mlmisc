import operator
import os
import re

import numpy as np
import py_misc_utils.alog as alog
import py_misc_utils.inspect_utils as pyiu
import py_misc_utils.module_utils as pymu
import py_misc_utils.utils as pyu
import torch
import torch.nn as nn
import torch.optim as optim

from .lrsched import reduce_on_plateau as rop


def _config_split(config):
  parts = pyu.resplit(config, ':')
  mod_config = pyu.parse_dict(parts[1], allow_args=True) if len(parts) == 2 else (dict(), ())

  return parts[0], mod_config


def _load_class(obj_name):
  m = re.match(r'(.*),([^\.]+)$', obj_name)
  if m and os.path.isfile(m.group(1)):
    module = pymu.import_module(m.group(1))
    obj_class = getattr(module, m.group(2))
  else:
    obj_class = operator.attrgetter(obj_name)(pyiu.current_module())

  return obj_class


def create_object(name, config, *args, **kwargs):
  obj_name, (obj_config, obj_args) = _config_split(config)

  kwargs.update(obj_config)

  alog.debug(f'Creating {obj_name} {name} with: ({len(args)} API args) {obj_args} {kwargs}')

  obj_class = _load_class(obj_name)

  return obj_class(*(args + obj_args), **kwargs)


def create_optimizer(params, config, **kwargs):
  return create_object('optimizer', config, params, **kwargs)


def create_lr_scheduler(optimizer, config, **kwargs):
  return create_object('LR scheduler', config, optimizer, **kwargs)


def create_loss(config, **kwargs):
  return create_object('Loss', config, **kwargs)


def create_model(config, *args, **kwargs):
  return create_object('Model', config, *args, **kwargs)

