import os

import numpy as np
import py_misc_utils.alog as alog
import py_misc_utils.assert_checks as tas
import py_misc_utils.utils as pyu
import torch
import torch.nn as nn
import torch.optim as optim

from .. import config as conf
from .. import core_utils as cu
from .. import utils as ut

from . import env as rlenv
from . import memory as rlmem
from . import nets as rlnets
from . import train_context as rltc
from . import utils as rlut


def create_env(model_name,
               model_path,
               env_args=None,
               device=None,
               q_optim='torch.optim.AdamW:lr=1.e-4',
               pi_optim='torch.optim.AdamW:lr=1.e-5',
               loss='torch.nn.HuberLoss',
               stepmem_size=100000,
               stepmem_dtype='float16',
               target_reward=100,
               action_noise_base=0.5,
               action_noise_end=0.05,
               tblog_dir=None,
               **kwargs):
  env_kwargs = pyu.parse_dict(env_args) if env_args else dict()
  env = rlenv.GymEnv(model_name, **env_kwargs)

  alog.info(f'Actions: {env.num_actions()}')
  alog.info(f'States: {env.num_signals()}')

  env.reset()
  init_screen = env.get_screen()
  channels, screen_height, screen_width = init_screen.shape

  alog.info(f'Screen: {screen_width}x{screen_height} ({channels} colors)')

  state_shape = (env.num_signals(),)
  pi_net = rlnets.PiNet(state_shape, env.num_actions()).to(device)
  q_net = rlnets.DRLN(state_shape, env.num_actions()).to(device)

  alog.info(f'PI Net (size={cu.net_memory_size(pi_net) * 1e-6:.1f} MB):\n{pi_net}')
  alog.info(f'Q Net (size={cu.net_memory_size(q_net) * 1e-6:.1f} MB):\n{q_net}')

  q_optimizer = conf.create_optimizer(q_net.parameters(), q_optim)
  pi_optimizer = conf.create_optimizer(pi_net.parameters(), pi_optim)

  q_lossfn = conf.create_loss(loss)

  memory = train_context = None
  if os.path.exists(model_path):
    data = ut.load_data(model_path,
                        strict=False,
                        q_net=q_net,
                        pi_net=pi_net)

    memory = data['memory']
    if memory.capacity() != stepmem_size:
      memory.resize(stepmem_size)

    train_context = data['train_context']
    train_context.reset(target_reward=target_reward,
                        action_noise_base=action_noise_base,
                        action_noise_end=action_noise_end)
  else:
    mem_fields = dict(state=env.num_signals(),
                      action=env.num_actions(),
                      next_state=env.num_signals(),
                      reward=1,
                      total_reward=1,
                      done=1)
    memory = rlmem.Memory(mem_fields, stepmem_size, dtype=np.dtype(stepmem_dtype))

    train_context = rltc.TrainContext(target_reward,
                                      action_noise_base=action_noise_base,
                                      action_noise_end=action_noise_end)
    data = None

  pi_net.train()
  q_net.train()

  saved_nets = dict(pi_net=pi_net, q_net=q_net)

  if tblog_dir is not None:
    stat_writer = cu.create_tb_writer(tblog_dir)
  else:
    stat_writer = cu.NoopTbWriter()

  return pyu.make_object(env=env,
                         q_net=q_net,
                         pi_net=pi_net,
                         checkpoint_data=data,
                         q_optimizer=q_optimizer,
                         pi_optimizer=pi_optimizer,
                         q_lossfn=q_lossfn,
                         saved_nets=saved_nets,
                         memory=memory,
                         train_context=train_context,
                         stat_writer=stat_writer)
