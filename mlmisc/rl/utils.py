import contextlib
import dataclasses
import functools
import itertools
import multiprocessing
import os
import queue
import time

import cv2
import numpy as np
import py_misc_utils.alog as alog
import py_misc_utils.app_main as app_main
import py_misc_utils.assert_checks as tas
import py_misc_utils.break_control as pybc
import py_misc_utils.core_utils as pycu
import py_misc_utils.fs_utils as pyfsu
import py_misc_utils.num_utils as pynu
import py_misc_utils.utils as pyu
import torch

from .. import core_utils as cu
from .. import debug_utils as du

from . import env as rlenv


def optimize_step(optimizer, loss, gclamp=None):
  optimizer.zero_grad()
  loss.backward()
  if gclamp is not None:
    cu.clamp_gradients(cu.optimizer_params(optimizer), gclamp)

  optimizer.step()


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
  alog.info(f'Train Perf ({nsteps} batches): {perf:.2e} batch/s ' \
            f'({perf * batch_size:.2e} sample/s)')

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


def net_infer(net, x, device=torch.device('cpu')):
  with cu.Training(net, False), torch.no_grad():
    x = torch.tensor(x).to(device)
    y = net(x)

    return y.numpy(force=True)


def select_action(envs, pi_eval, state,
                  noise_sigma=0.0,
                  mode='train',
                  needs_batch=True):
  tas.check(mode in {'train', 'infer'}, msg=f'Unknown mode: {mode}')

  if needs_batch:
    bstate = np.expand_dims(state, axis=0)
    baction = pi_eval(bstate)
    action = np.squeeze(baction, axis=0)
  else:
    action = pi_eval(state)

  if mode == 'infer':
    return action

  if needs_batch:
    return envs[0].rand(noise_sigma, action=action)

  return [envs[i].rand(noise_sigma, action=act) for i, act in enumerate(action)]


@dataclasses.dataclass
class Sample:
  state: np.ndarray
  action: np.ndarray
  next_state: np.ndarray
  reward: float
  done: float
  total_reward: float = None


def run_episodes(env, pi_net, count,
                 batch_size=64,
                 noise_sigma=0.0,
                 device=None,
                 final_reward=None,
                 max_episode_steps=1000):
  batch_size = min(count, batch_size)

  pi_eval = functools.partial(net_infer, pi_net, device=device)

  envs = [env.new() for _ in range(batch_size)]
  states = [envs[i].reset() for i in range(batch_size)]
  running = list(range(batch_size))
  samples = [[] for _ in range(batch_size)]
  traces = []
  while running:
    actions = select_action([envs[idx] for idx in running], pi_eval, np.vstack(states),
                            noise_sigma=noise_sigma,
                            needs_batch=False)

    next_states, next_running = [], []
    for i in range(len(actions)):
      idx = running[i]
      next_state, reward, done = envs[idx].step(actions[i])

      if len(samples[idx]) + 1 >= max_episode_steps:
        done = rlenv.TERMINATED
      if done != rlenv.ALIVE and final_reward is not None:
        step_reward = final_reward
      else:
        step_reward = reward

      samples[idx].append(Sample(states[i], actions[i], next_state, step_reward, done))

      if done != rlenv.ALIVE:
        traces.append(samples[idx])
        samples[idx] = []

        if count > len(traces) + len(next_running):
          state = envs[idx].reset()
          next_states.append(state)
          next_running.append(idx)
      else:
        next_states.append(next_state)
        next_running.append(idx)

    states, running = next_states, next_running

  results = []
  for trace in traces:
    total_reward = episode_reward = sum(sample.reward for sample in trace)
    for sample in trace:
      sample.total_reward = total_reward
      total_reward -= sample.reward

    result = pyu.make_object(step_count=len(trace),
                             episode_reward=episode_reward,
                             samples=trace)
    results.append(result)

  return results


def _mp_run_episodes(pidx, rqueue, env, pi_net, count=1, **kwargs):
  try:
    with pybc.BreakControl() as bc:
      rqueue.put((pidx, None))

      results = run_episodes(env, pi_net, count, **kwargs)

      rqueue.put((pidx, results))
  except Exception as ex:
    rqueue.put((pidx, ex))


def _collect_mp_results(rqueue, workers):
  results = []
  exceptions, to_ack = [], set(workers.keys())
  while workers:
    try:
      qvalue = rqueue.get(True, 0.5)
    except queue.Empty:
      qvalue = None

    if qvalue is not None:
      pidx, value = qvalue
      if value is None:
        to_ack.discard(pidx)
      elif isinstance(value, Exception):
        exceptions.append(value)
      else:
        results.extend(value)

        workers[pidx].join()
        workers.pop(pidx)

    for pidx in tuple(to_ack):
      if not workers[pidx].is_alive():
        workers[pidx].join()
        workers.pop(pidx)
        to_ack.discard(pidx)

  if exceptions:
    xmsg = '\n'.join(f'[{i}] {ex}' for i, ex in enumerate(exceptions))
    alog.error(f'Exception in learner worker:\n{xmsg}')
    raise type(exceptions[0])(xmsg)

  return results


_MIN_PERCPU_EPISODES = int(os.getenv('MIN_PERCPU_EPISODES', 50))

def learn(env, pi_net, memory,
          num_episodes=1,
          num_workers=None,
          **kwargs):
  num_workers = num_workers or os.cpu_count()

  num_workers = min(num_workers, num_episodes // _MIN_PERCPU_EPISODES)

  results = []
  if num_workers <= 1:
    results.extend(run_episodes(env, pi_net, num_episodes, **kwargs))
  else:
    worker_kwargs = kwargs.copy()
    worker_kwargs.update(count=round(num_episodes / num_workers))

    mpctx = multiprocessing.get_context('spawn')

    with contextlib.closing(mpctx.Queue()) as rqueue:
      workers = dict()
      for i in range(num_workers):
        worker = app_main.create_process(_mp_run_episodes,
                                         args=(i, rqueue, env, pi_net),
                                         kwargs=worker_kwargs,
                                         context=mpctx,
                                         daemon=True)
        worker.start()
        workers[i] = worker

      results.extend(_collect_mp_results(rqueue, workers))

  total_steps = sum(r.step_count for r in results)
  avg_nsteps = total_steps // len(results)
  avg_reward = np.mean(tuple(r.episode_reward for r in results)).item()

  for result in results:
    for sample in result.samples:
      memory.append(state=sample.state,
                    action=sample.action,
                    next_state=sample.next_state,
                    reward=np.array([sample.reward]),
                    total_reward=np.array([sample.total_reward]),
                    done=np.array([sample.done]))

  return pyu.make_object(total_steps=total_steps,
                         avg_nsteps=avg_nsteps,
                         avg_reward=avg_reward)


def make_video(path, env, pi_net,
               device=None,
               max_episode_steps=1000,
               fps=10):
  pi_eval = functools.partial(net_infer, pi_net, device=device)
  tmp_path = pyfsu.temp_path(nspath=path)

  fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

  total_reward, out = 0, None
  state = env.reset()
  with contextlib.ExitStack() as xstack:
    for stepno in itertools.count():
      action = select_action([env], pi_eval, state, mode='infer')
      next_state, reward, done = env.step(action)
      total_reward += reward

      screen = env.get_screen().transpose((1, 2, 0))
      if out is None:
        height, width, channels = screen.shape
        alog.debug(f'Generating video ({width}x{height} {channels} colors) to {path}')
        out = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height), isColor=True)
        xstack.callback(out.release)

      screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
      screen = cv2.normalize(screen, None, 255, 0,
                             norm_type=cv2.NORM_MINMAX,
                             dtype=cv2.CV_8U)
      out.write(screen)

      if done != rlenv.ALIVE:
        break
      state = next_state
      if stepno >= max_episode_steps:
        alog.info(f'Too many steps ({stepno}) ... aborting episode')
        break

    alog.debug(f'Video generated in {stepno} steps, reward was {total_reward:.2e}')

  os.replace(tmp_path, path)

