import collections
import contextlib
import functools
import itertools
import os
import time

import cv2
import numpy as np
import py_misc_utils.alog as alog
import py_misc_utils.assert_checks as tas
import py_misc_utils.core_utils as pycu
import py_misc_utils.fs_utils as pyfsu
import py_misc_utils.num_utils as pynu
import py_misc_utils.utils as pyu
import torch

from .. import core_utils as cu
from .. import debug_utils as du


def optimize_step(optimizer, loss, gclamp=None):
  optimizer.zero_grad()
  loss.backward()
  if gclamp is not None:
    cu.clamp_gradients(cu.optimizer_params(optimizer), gclamp)

  optimizer.step()


def select_action(env, train_context, net, state,
                  device=None,
                  mode='train'):
  tas.check(mode in {'train', 'infer'}, msg=f'Unknown mode: {mode}')

  with cu.Training(net, False), torch.no_grad():
    x = torch.tensor(state).unsqueeze(0)
    x = x if device is None else x.to(device)
    action = net(x)

  action = action.squeeze(0).numpy(force=True)

  if mode == 'train':
    noise_sigma = train_context.action_noise()
    action = env.rand(noise_sigma, action=action)

  return action


# This, like the LinearParamUpdater, is kept around for historical reasons,
# since the run_step_trew() works much better in every environment I tested
# them with.
def qlearn_step(q_net,
                target_q_net,
                pi_net,
                q_optimizer,
                pi_optimizer,
                q_lossfn,
                rec,
                gamma=0.95,
                qnet_gclamp=None,
                pnet_gclamp=None):
  q_values = q_net(rec.state, rec.action)

  with torch.no_grad():
    next_q = target_q_net(rec.next_state, pi_net(rec.next_state))
    q_prime = next_q * gamma * (1.0 - rec.done.abs()) + rec.reward

  q_loss = q_lossfn(q_values, q_prime.detach())

  optimize_step(q_optimizer, q_loss, gclamp=qnet_gclamp)

  pi_action = pi_net(rec.state)
  pi_loss = -target_q_net(rec.state, pi_action).mean()

  target_q_net.zero_grad()
  optimize_step(pi_optimizer, pi_loss, gclamp=pnet_gclamp)

  return q_loss.item(), pi_loss.item()


def trew_qlearn_step(q_net,
                     target_q_net,
                     pi_net,
                     q_optimizer,
                     pi_optimizer,
                     q_lossfn,
                     rec,
                     gamma=0.95,
                     qnet_gclamp=None,
                     pnet_gclamp=None):
  q_values = q_net(rec.state, rec.action)

  with torch.no_grad():
    next_q = target_q_net(rec.state, rec.action)
    q_prime = pynu.mix(next_q, rec.total_reward, gamma)

  q_loss = q_lossfn(q_values, q_prime.detach())

  optimize_step(q_optimizer, q_loss, gclamp=qnet_gclamp)

  pi_action = pi_net(rec.state)
  pi_loss = -target_q_net(rec.state, pi_action).mean()

  target_q_net.zero_grad()
  optimize_step(pi_optimizer, pi_loss, gclamp=pnet_gclamp)

  return q_loss.item(), pi_loss.item()


def trew_step(q_net,
              pi_net,
              q_optimizer,
              pi_optimizer,
              q_lossfn,
              rec,
              gamma=0.9,
              qnet_gclamp=None,
              pnet_gclamp=None):
  q_values = q_net(rec.state, rec.action)
  q_prime = pynu.mix(q_values, rec.total_reward, gamma)

  q_loss = q_lossfn(q_values, q_prime.detach())

  optimize_step(q_optimizer, q_loss, gclamp=qnet_gclamp)

  pi_action = pi_net(rec.state)
  pi_loss = -q_net(rec.state, pi_action).mean()

  q_net.zero_grad()
  optimize_step(pi_optimizer, pi_loss, gclamp=pnet_gclamp)

  return q_loss.item(), pi_loss.item()


def sample_trans(v, dtype=np.float32, device=None):
  v = v.astype(dtype)
  v = torch.tensor(v)

  return v if device is None else v.to(device)


def optimize_model(memory, batch_size, stepfn, device=None, nsteps=1, bc=None):
  tstart = time.time()

  trans = functools.partial(sample_trans, device=device)
  rec_iter = memory.iter_samples(batch_size, trans=trans)

  q_losses, pi_losses = [], []
  for n in range(nsteps):
    if (rec := pycu.iter_next(rec_iter)) is None:
      break

    q_loss, pi_loss = stepfn(rec)

    q_losses.append(q_loss)
    pi_losses.append(pi_loss)
    if bc is not None and bc.hit():
      break

  perf = (n + 1) / (time.time() - tstart)
  alog.info(f'Train Perf ({nsteps} batches): {perf:.2e} batch/s')

  return q_losses, pi_losses


def write_tensor_stats(name_fmt, stats, train_context, stat_writer):
  for ts in stats:
    value = {k: getattr(ts, k)
             for k in pyu.comma_split('min, max, mean, std')}
    for p, pv in zip(ts.percentiles, ts.percentile_values):
      value[f'p{int(100 * p)}'] = pv

    cu.tb_write(stat_writer, name_fmt.format(ts.name), value, train_context.stepno)


def show_grad_info(net, train_context, stat_writer,
                   percentiles=(0.5, 0.9, 0.99)):
  pstats = du.get_parameters_stats(net, percentiles=percentiles, fmt='.2e')
  du.show_tensors_stats(pstats, dict(value_stats=alog.DEBUG))

  gstats = du.get_grads_stats(net, percentiles=percentiles, fmt='.2e')
  du.show_tensors_stats(gstats, dict(value_stats=alog.DEBUG))

  write_tensor_stats('PARA.{}', pstats.stats, train_context, stat_writer)
  write_tensor_stats('GRAD.{}', gstats.stats, train_context, stat_writer)


def show_param_divergence(net, tnet, train_context, stat_writer):
  tparams = {n: p for n, p in tnet.named_parameters()}
  for name, param in net.named_parameters():
    tparam = tparams[name]
    pdiff = torch.abs(param.data - tparam.data).mean()
    pweight = torch.maximum(torch.abs(param.data), torch.abs(tparam.data)).mean()
    pdiv = (pdiff / pweight).item()

    cu.tb_write(stat_writer, f'PDiv({name})', pdiv, train_context.stepno)
    alog.info(f'\t{name} {tuple(param.shape)} = {pdiv:.2e}')


def show_context(ctx):
  alog.debug(f'Parsed Args:')
  for k, v in vars(ctx).items():
    alog.debug(f'  {k} = {v}')


def show_samples(memory, percentiles=(0.5, 0.8, 0.9, 0.99), dest_path=None):
  alog.info(f'Samples Info:')
  df = memory.dataframe()
  if len(df) > 0:
    for col in df.columns:
      destr = str(df[col].describe(percentiles=percentiles))
      dlines = [f'  {ln}' for ln in destr.splitlines()]
      dlines.append('')

      alog.info('\n'.join(dlines))

    if dest_path is not None:
      df.to_pickle(dest_path)


def make_video(path, env, train_context, pi_net,
               device=None,
               max_episode_steps=1000,
               fps=10):
  tmp_path = pyfsu.temp_path(nspath=path)

  fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

  total_reward, out = 0, None
  state = env.reset()
  with contextlib.ExitStack() as xstack:
    for t in itertools.count():
      action = select_action(env, train_context, pi_net, state,
                             device=device,
                             mode='infer')
      next_state, reward, done = env.step(action)
      total_reward += reward

      screen = env.get_screen().transpose((1, 2, 0))
      if out is None:
        height, width, colors = screen.shape
        alog.debug(f'Generating video ({width}x{height} {colors} colors) to {path}')
        out = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height), isColor=True)
        xstack.callback(out.release)

      screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
      screen = cv2.normalize(screen, None, 255, 0,
                             norm_type=cv2.NORM_MINMAX,
                             dtype=cv2.CV_8U)
      out.write(screen)

      if done != env.ALIVE:
        break
      state = next_state
      if t >= max_episode_steps:
        alog.info(f'Too many steps ({t}) ... aborting episode')
        break

    alog.debug(f'Video generated in {t} steps, reward was {total_reward:.2e}')

  os.replace(tmp_path, path)


Sample = collections.namedtuple(
  'Sample',
  'state, action, next_state, reward, done')

def run_episode(env, train_context, pi_net, memory,
                device=None,
                final_reward=None,
                max_episode_steps=1000):
  samples = []
  state = env.reset()
  for ep_step in itertools.count():
    action = select_action(env, train_context, pi_net, state, device=device)

    next_state, reward, done = env.step(action)

    if ep_step >= max_episode_steps:
      alog.info(f'Too many steps ({ep_step}) ... aborting episode')
      done = env.TERMINATED

    step_reward = final_reward if done != env.ALIVE and final_reward is not None else reward

    samples.append(Sample(state, action, next_state, step_reward, done))
    state = next_state

    if done != env.ALIVE:
      break

  total_reward = episode_reward = sum(s.reward for s in samples)
  for s in samples:
    memory.append(state=s.state,
                  action=s.action,
                  next_state=s.next_state,
                  reward=[s.reward],
                  total_reward=[total_reward],
                  done=[s.done])
    total_reward -= s.reward

  return pyu.make_object(step_count=ep_step,
                         episode_reward=episode_reward)

