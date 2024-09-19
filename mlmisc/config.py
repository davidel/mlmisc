import operator

import numpy as np
import py_misc_utils.alog as alog
import py_misc_utils.utils as pyu
import torch
import torch.nn as nn
import torch.optim as optim

from .lrsched import reduce_on_plateau as rop


def _config_split(config):
  parts = pyu.resplit(config, ':')
  mod_config = pyu.parse_dict(parts[1], allow_args=True) if len(parts) == 2 else (dict(), [])

  return parts[0], mod_config


def create_optimizer(params, config, **kwargs):
  optim_name, (optim_config, optim_args) = _config_split(config)
  kwargs.update(optim_config)

  alog.debug(f'Creating {optim_name} optimizer with: {optim_args} {kwargs}')

  optim_class = operator.attrgetter(optim_name)(pyu.current_module())

  return optim_class(params, *optim_args, **kwargs)


def create_lr_scheduler(optimizer, config, **kwargs):
  sched_name, (sched_config, sched_args) = _config_split(config)
  kwargs.update(sched_config)

  alog.debug(f'Creating {sched_name} LR scheduler with: {sched_args} {kwargs}')

  sched_class = operator.attrgetter(sched_name)(pyu.current_module())

  return sched_class(optimizer, *sched_args, **kwargs)

