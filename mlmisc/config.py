import operator
import sys

import numpy as np
import py_misc_utils.alog as alog
import py_misc_utils.utils as pyu
import torch
import torch.nn as nn
import torch.optim as optim


def create_optimizer(params, config):
  optim_name, optim_config = pyu.resplit(config, ':')
  optim_config = pyu.parse_dict(optim_config)

  alog.debug(f'Creating {optim_name} optimizer with: {optim_config}')

  optim_class = operator.attrgetter(optim_name)(sys.modules[__name__])

  return optim_class(params, **optim_config)


def create_lr_scheduler(optimizer, config):
  sched_name, sched_config = pyu.resplit(config, ':')
  sched_config = pyu.parse_dict(sched_config)

  alog.debug(f'Creating {sched_name} LR scheduler with: {sched_config}')

  sched_class = operator.attrgetter(sched_name)(sys.modules[__name__])

  return sched_class(optimizer, **sched_config)

