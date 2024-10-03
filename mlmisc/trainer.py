import array
import datetime
import time

import numpy as np
from py_misc_utils import alog
from py_misc_utils import utils as pyu
import torch
import torch.nn as nn

from . import debug_utils as du
from . import utils as ut


class TimeTracker:

  def __init__(self, total=0):
    self.stamp = None
    self.total = total if isinstance(total, datetime.timedelta) else datetime.timedelta(seconds=total)

  def start(self):
    self.stamp = time.time()

    return self.stamp

  def track(self):
    now = time.time()
    self.total += datetime.timedelta(seconds=now - self.stamp)
    self.stamp = now

    return now

  @property
  def seconds(self):
    return self.total.total_seconds()


class LrScheduler:

  def __init__(self, scheduler):
    self.scheduler = scheduler

  def train_step(self, loss):
    if self.scheduler is not None:
      train_step = getattr(self.scheduler, 'train_step', None)
      if train_step is not None:
        return train_step(loss)

  def epoch_step(self, loss):
    if self.scheduler is not None:
      epoch_step = getattr(self.scheduler, 'epoch_step', None)
      if epoch_step is not None:
        return epoch_step(loss)
      else:
        self.scheduler.step()
        alog.debug(f'Scheduler step: lr={pyu.format(self.scheduler.get_last_lr(), ".3e")}')


class Trainer:

  def __init__(self):
    self._load_state(dict())

  def _load_state(self, state):
    self._num_samples = state.get('num_samples', 0)
    self._train_time = TimeTracker(total=state.get('train_time', 0))
    self._val_time = TimeTracker(total=state.get('val_time', 0))
    self._save_time = TimeTracker(total=state.get('save_time', 0))

  def _get_state(self):
    return dict(
      num_samples=self._num_samples,
      train_time=self._train_time.total,
      val_time=self._val_time.total,
      save_time=self._save_time.total,
    )

  def save_model(self, model, path, **kwargs):
    self._save_time.start()
    state = self._get_state()
    ut.save_data(path, model=model, **kwargs, **state)
    self._save_time.track()

  def load_model(self, path, device=None, strict=True):
    state = ut.load_data(path, strict=strict)
    self._load_state(state)

    model = state['model']
    if device is not None:
      model = model.to(device)

    return model, state

  def load_aux_state(self, state, **kwargs):
    loaded = []
    for name, obj in kwargs.items():
      ostate = state.get(name)
      if ostate is not None:
        alog.debug(f'Loading "{name}" state')
        obj.load_state_dict(ostate)
        loaded.append(name)

    return tuple(loaded)

  def _times(self):
    return f'train={self._train_time.total}\t' \
      f'val={self._val_time.total}\t' \
      f'save={self._save_time.total}'

  def _val_loss(self, tctx):
    loader = torch.utils.data.DataLoader(tctx.val_data,
                                         batch_size=tctx.batch_size or 1,
                                         shuffle=True,
                                         num_workers=tctx.num_workers)

    alog.info(f'Running validation on {len(loader)} batches')

    tctx.model.eval()
    with torch.no_grad():
      losses, val_start = [], self._val_time.start()
      for i, (x, y) in enumerate(loader):
        if tctx.device is not None:
          x, y = x.to(tctx.device), y.to(tctx.device)

        _, loss = tctx.model(x, targets=y)
        losses.append(loss.item())

        if ((tctx.val_time is not None and time.time() > val_start + tctx.val_time) or
            (callable(tctx.should_stop) and tctx.should_stop())):
          alog.info(f'Validation run on {i} of {len(loader)} batches due to ' \
                    f'{datetime.timedelta(seconds=tctx.val_time)} required time stop')
          break

    tctx.model.train()

    self._val_time.track()

    return np.mean(losses) if losses else None

  def _tblog(self, tb_writer, total_samples, name, value):
    epoch = 100 * self._num_samples / total_samples
    if tb_writer is not None:
      tb_writer.add_scalar(name, value, global_step=int(epoch * 10))

    return epoch

  def _log_train_loss(self, loss, batch_num, num_batches, step_time, tctx):
    epoch = self._tblog(tctx.tb_writer, num_batches * tctx.batch_size, 'Train Loss', loss)
    alog.info(f'Batch {batch_num + 1}/{num_batches} (epoch={epoch:.1f}%): ' \
              f'Train Loss {loss:.4f}')
    alog.info(f'Times: {self._times()}')
    alog.info(f'Perf: {tctx.batch_size / step_time:.2e} samples/sec')

  def _run_validation(self, batch_num, num_batches, tctx):
    vloss = self._val_loss(tctx)
    if vloss is not None:
      epoch = self._tblog(tctx.tb_writer, num_batches * tctx.batch_size,
                          'Validation Loss', vloss)
      alog.info(f'Batch {batch_num + 1}/{num_batches} (epoch={epoch:.1f}%): ' \
                f'Validation Loss {vloss:.4f}')

    return vloss

  def _show_stats(self, model):
    percentiles = (0.5, 0.9, 0.95, 0.99)
    du.show_tensors_stats(du.get_parameters_stats(model,
                                                  percentiles=percentiles),
                          dict(value_stats=alog.DEBUG))
    du.show_tensors_stats(du.get_grads_stats(model,
                                             percentiles=percentiles),
                          dict(value_stats=alog.DEBUG))

  def _save_checkpoint(self, tctx):
    checkpoint = tctx.checkpoint or ('optimizer', 'scheduler', 'scaler')
    cargs = dict()
    for name in checkpoint:
      data = getattr(tctx, name, None)
      if data is not None:
        cargs[name] = data

    self.save_model(tctx.model, tctx.model_path, **cargs)

  def _step(self, tctx):
    loader = torch.utils.data.DataLoader(tctx.train_data,
                                         batch_size=tctx.batch_size,
                                         shuffle=True,
                                         num_workers=tctx.num_workers)

    num_batches = len(loader)
    alog.info(f'Running EPOCH train on {num_batches} batches')

    tctx.model.train()
    tctx.optimizer.zero_grad()

    for i, (x, y) in enumerate(loader):
      if tctx.device is not None:
        x, y = x.to(tctx.device), y.to(tctx.device)

      if tctx.scaler is not None:
        with torch.autocast(device_type=tctx.device.type,
                            dtype=tctx.amp_dtype or torch.float16):
          _, loss = tctx.model(x, targets=y)
      else:
        _, loss = tctx.model(x, targets=y)

      bloss = loss if tctx.accum_steps == 1 else loss / tctx.accum_steps

      if tctx.scaler is not None:
        tctx.scaler.scale(bloss).backward()
      else:
        bloss.backward()

      self._num_samples += tctx.batch_size
      if (i + 1) % tctx.accum_steps == 0:
        if tctx.scaler is not None:
          if tctx.grad_clip is not None and tctx.grad_clip > 0:
            tctx.scaler.unscale_(tctx.optimizer)
            nn.utils.clip_grad_norm_(tctx.model.parameters(), tctx.grad_clip)

          tctx.scaler.step(tctx.optimizer)
          tctx.scaler.update()
        else:
          if tctx.grad_clip is not None and tctx.grad_clip > 0:
            nn.utils.clip_grad_norm_(tctx.model.parameters(), tctx.grad_clip)

          tctx.optimizer.step()

        yield pyu.make_object(stepno=i, loss=loss, num_batches=num_batches)

        tctx.optimizer.zero_grad()

  def train_epoch(self, model, optimizer, train_data, val_data, batch_size,
                  device=None,
                  scheduler=None,
                  accum_steps=1,
                  grad_clip=None,
                  val_time=None,
                  loss_logstep=60,
                  val_logstep=900,
                  model_chkptstep=600,
                  checkpoint=None,
                  model_path=None,
                  tb_writer=None,
                  num_workers=0,
                  should_stop=None,
                  step_fn=None,
                  scaler=None,
                  amp_dtype=None):
    tctx = pyu.make_object(**{k: v for k, v in locals().items() if k != 'self'})

    wrapped_scheduler = LrScheduler(scheduler)

    tstep, tval, tsave = [self._train_time.start()] * 3
    train_losses, val_losses, last_stepno = array.array('f'), array.array('f'), -1
    for sd in self._step(tctx):
      now = self._train_time.track()
      if now > tstep + loss_logstep:
        train_losses.append(sd.loss.item())
        self._log_train_loss(train_losses[-1], sd.stepno, sd.num_batches,
                             (now - tstep) / (sd.stepno - last_stepno), tctx)
        tstep, last_stepno = now, sd.stepno

      wrapped_scheduler.train_step(sd.loss.item())

      if model_path is not None and now > tsave + model_chkptstep:
        self._train_time.track()
        self._save_checkpoint(tctx)
        tsave = self._train_time.start()

      if now > tval + val_logstep or sd.stepno + accum_steps >= sd.num_batches:
        self._train_time.track()
        self._show_stats(model)
        vloss = self._run_validation(sd.stepno, sd.num_batches, tctx)
        if vloss is not None:
          val_losses.append(vloss)
        tval = self._train_time.start()

      if step_fn is not None:
        step_fn(sd)

      stopped = callable(should_stop) and should_stop()
      if stopped:
        alog.info(f'Interrupted at batch {sd.stepno + 1}/{sd.num_batches}!')
        break

    optimizer.step()

    if not stopped:
      wrapped_scheduler.epoch_step(np.mean(val_losses) if val_losses else None)

    self._train_time.track()

    if model_path is not None:
      self._save_checkpoint(tctx)

    return train_losses, val_losses

