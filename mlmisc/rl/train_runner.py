import functools
import itertools
import os
import time
import typing

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import py_misc_utils.alog as alog
import py_misc_utils.app_main as app_main
import py_misc_utils.assert_checks as tas
import py_misc_utils.break_control as pybc
import py_misc_utils.core_utils as pycu
import py_misc_utils.inspect_utils as pyiu
import py_misc_utils.time_trigger as pytt
import py_misc_utils.utils as pyu
import torch

from .. import core_utils as cu
from .. import utils as ut

from . import env as rlenv
from . import main_utils as rlmu
from . import nets as rlnets
from . import param_updater as rlpu
from . import utils as rlut


def _create_param_updater(ctx, env):
  if (target_q_net := getattr(env, 'target_q_net', None)) is not None:
    if ctx.param_updater == 'checkpoint':
      return rlpu.CheckpointParamUpdater(env.q_net,
                                         target_q_net,
                                         ctx.trains_per_update,
                                         device=ctx.device)
    if ctx.param_updater == 'linear':
      return rlpu.LinearParamUpdater(env.q_net,
                                     target_q_net,
                                     ctx.trains_per_update,
                                     ctx.tau)

  if ctx.param_updater == 'noop':
    return rlpu.NoopParamUpdater()

  alog.xraise(ValueError, f'Unknown parameter updater type: {ctx.param_updater}')


def _show_statistics(env):
  alog.info(f'Q_NET Param/Grad Info:')
  rlut.show_grad_info(env.q_net, env.train_context, env.stat_writer)

  alog.info(f'PI_NET Param/Grad Info:')
  rlut.show_grad_info(env.pi_net, env.train_context, env.stat_writer)

  if (target_q_net := getattr(env, 'target_q_net', None)) is not None:
    alog.info(f'Q_NET Divergence:')
    rlut.show_param_divergence(env.q_net, target_q_net, env.train_context,
                               env.stat_writer)

  alog.info(f'Num Samples: {len(env.memory)}')
  alog.info(f'RndEps: {env.train_context.action_noise():.2e}')


def _create_env(ctx):
  env = pyiu.fetch_call(rlmu.create_env, vars(ctx))

  if ctx.arch == 'trew':
    env.stepfn = functools.partial(rlut.trew_step,
                                   env.q_net,
                                   env.pi_net,
                                   env.q_optimizer,
                                   env.pi_optimizer,
                                   env.q_lossfn,
                                   gamma=ctx.gamma,
                                   qnet_gclamp=ctx.qnet_gclamp,
                                   pnet_gclamp=ctx.pnet_gclamp)
  elif ctx.arch in {'qlearn', 'trew_qlearn'}:
    env.target_q_net = rlnets.DRLN(env.env.num_signals(), env.env.num_actions()).to(ctx.device)
    if env.checkpoint_data is not None and 'target_q_net' in env.checkpoint_data:
      ut.load_state(env.checkpoint_data, target_q_net=env.target_q_net)
    else:
      cu.update_params(env.q_net, env.target_q_net)

    env.target_q_net.train()
    env.saved_nets.update(target_q_net=env.target_q_net)

    stepfn = rlut.qlearn_step if ctx.arch == 'qlearn' else rlut.trew_qlearn_step
    env.stepfn = functools.partial(stepfn,
                                   env.q_net,
                                   env.target_q_net,
                                   env.pi_net,
                                   env.q_optimizer,
                                   env.pi_optimizer,
                                   env.q_lossfn,
                                   gamma=ctx.gamma,
                                   qnet_gclamp=ctx.qnet_gclamp,
                                   pnet_gclamp=ctx.pnet_gclamp)
  else:
    alog.xraise(ValueError, f'Unknown step mode: {ctx.arch}')

  # Drop to save memory.
  delattr(env, 'checkpoint_data')

  return env


def _train_loop(ctx, env):
  bc = pybc.create()

  show_episode = pytt.TimeTrigger(ctx.show_episode_nsecs)
  print_screen = pytt.TimeTrigger(ctx.print_screen_nsecs)
  do_checkpoint = pytt.TimeTrigger(ctx.checkpoint_nsecs)
  show_stats = pytt.TimeTrigger(ctx.stats_print_nsecs)

  refill_train_size = int(env.memory.capacity() * ctx.train_capacity_pct)

  param_updater = _create_param_updater(ctx, env)

  pushed, ep_results = 0, []
  for e in range(ctx.num_episodes):
    with env.train_context.sampling():
      epres = rlut.run_episode(env.env, env.pi_net, env.memory,
                               noise_sigma=env.train_context.action_noise(),
                               device=ctx.device,
                               final_reward=ctx.final_reward,
                               max_episode_steps=ctx.max_episode_steps)

    pushed += epres.step_count
    ep_results.append(epres)

    cu.tb_write(env.stat_writer, 'Episode Reward', epres.episode_reward,
                env.train_context.stepno)
    cu.tb_write(env.stat_writer, 'Episode Steps', epres.step_count,
                env.train_context.stepno)
    if show_episode:
      avg_steps = np.mean([er.step_count for er in ep_results])
      cu.tb_write(env.stat_writer, 'AvgSteps', avg_steps, env.train_context.stepno)
      avg_reward = np.mean([er.episode_reward for er in ep_results])
      cu.tb_write(env.stat_writer, 'AvgReward', avg_reward, env.train_context.stepno)
      ep_results = []
      alog.info(f'[{e}/{env.train_context.stepno}] Steps {avg_steps:.1f}\tReward {avg_reward:.2e}\t')

    if pushed >= refill_train_size:
      alog.info(f'[{env.train_context.stepno}] SamplingTime = {env.train_context.sampling_time()}' \
                f'\tTrainingTime = {env.train_context.training_time()}')
      alog.info(f'[{env.train_context.stepno}] Training ({pushed} new samples) ...')

      pushed = 0
      num_steps = int(ctx.train_coverage_pct * len(env.memory) / ctx.batch_size)

      with env.train_context.training():
        q_losses, pi_losses = rlut.optimize_model(env.memory, ctx.batch_size, env.stepfn,
                                                  device=ctx.device,
                                                  nsteps=num_steps,
                                                  bc=bc.v)

      q_loss, pi_loss = np.mean(q_losses), np.mean(pi_losses)

      cu.tb_write(env.stat_writer, 'QLoss', q_loss, env.train_context.stepno)
      cu.tb_write(env.stat_writer, 'PiLoss', pi_loss, env.train_context.stepno)
      alog.info(f'[{env.train_context.stepno}] QLoss = {q_loss:.2e}\tPiLoss = {pi_loss:.2e}')

      env.train_context.train_step()
      param_updater.update(env.train_context.train_stepno)

      env.memory.filter(('state', 'action'), 1.0 - ctx.train_capacity_pct)

      if ctx.video_path is not None:
        rlut.make_video(ctx.video_path, env.env, env.train_context, env.pi_net,
                        device=ctx.device,
                        max_episode_steps=ctx.max_episode_steps)

    if show_stats:
      _show_statistics(env)

    if print_screen:
      env.env.print_screen(path=ctx.image_path)

    env.train_context.step(epres.episode_reward, epres.step_count)

    if do_checkpoint:
      ut.save_data(ctx.model_path,
                   **env.saved_nets,
                   memory=env.memory,
                   train_context=env.train_context)

    if bc.v.hit():
      break

  _show_statistics(env)
  ut.save_data(ctx.model_path,
               **env.saved_nets,
               memory=env.memory,
               train_context=env.train_context)
  env.stat_writer.close()


def train(model_name,
          model_path: str=None,
          env_args: str=None,
          arch: typing.Literal['trew', 'qlearn', 'trew_qlearn']='trew',
          batch_size=1024,
          q_optim='torch.optim.AdamW:lr=1.e-4',
          pi_optim='torch.optim.AdamW:lr=1.e-5',
          loss='torch.nn.HuberLoss',
          gamma=0.9,
          final_reward: float=None,
          target_reward=1500,
          action_noise_base=0.5,
          action_noise_end=0.05,
          qnet_gclamp: float=None,
          pnet_gclamp: float=None,
          num_episodes=50000,
          train_capacity_pct=0.1,
          train_coverage_pct=0.3,
          param_updater: typing.Literal['checkpoint', 'linear', 'noop']='checkpoint',
          tau=0.95,
          trains_per_update=5,
          stepmem_size=200000,
          stepmem_dtype='float16',
          stepmem_path: str=None,
          num_cpu_threads: int=None,
          show_episode_nsecs=4,
          print_screen_nsecs=0,
          checkpoint_nsecs=300,
          stats_print_nsecs=60,
          tblog_dir: str=None,
          image_path: str=None,
          video_path: str=None,
          max_episode_steps=1000):
  device = cu.get_device()
  if model_path is None:
    model_path = os.path.join(os.getcwd(), f'{model_name}.ckpt')
    alog.info(f'Using model path: {model_path}')

  ctx = pyu.locals_capture(locals())
  rlut.show_context(ctx)

  if num_cpu_threads is not None:
    torch.set_num_threads(num_cpu_threads)

  env = _create_env(ctx)

  rlut.show_samples(env.memory, dest_path=stepmem_path)

  _train_loop(ctx, env)


if __name__ == '__main__':
  app_main.basic_main(app_main.Main(train), description='RL Test')

