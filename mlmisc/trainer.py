import array
import datetime
import math
import time

import numpy as np
import py_misc_utils.alog as alog
import py_misc_utils.buffered_iterator as pybi
import py_misc_utils.utils as pyu
import torch
import torch.nn as nn

from . import core_utils as cu
from . import data_loader as dload
from . import dataset_utils as dsu
from . import debug_utils as du
from . import utils as ut
from .lrsched import wrapper as lrw


class TimeTracker:

  def __init__(self, total=0):
    self._stamp = None
    self.total = total if isinstance(total, datetime.timedelta) else datetime.timedelta(seconds=total)

  def start(self):
    self._stamp = time.time()

    return self._stamp

  def track(self):
    now = time.time()
    self.total += datetime.timedelta(seconds=now - self._stamp)
    self._stamp = now

    return now

  @property
  def seconds(self):
    return self.total.total_seconds()


class Trainer:

  def __init__(self):
    self.load_state(dict())

  def load_state(self, state):
    self.global_step = state.get('global_step', 0)
    self.num_samples = state.get('num_samples', 0)
    self.train_time = TimeTracker(total=state.get('train_time', 0))
    self.val_time = TimeTracker(total=state.get('val_time', 0))
    self.save_time = TimeTracker(total=state.get('save_time', 0))
    self.metrics = state.get('metrics', [])

  def get_state(self):
    return dict(
      global_step=self.global_step,
      num_samples=self.num_samples,
      train_time=self.train_time.total,
      val_time=self.val_time.total,
      save_time=self.save_time.total,
      metrics=self.metrics,
    )

  def save_model(self, model, path, **kwargs):
    self.save_time.start()
    state = self.get_state()
    ut.save_data(path, model=model, **kwargs, **state)
    self.save_time.track()

  def load_model(self, path, device=None, strict=True):
    model, state = self.load(path, device=device, strict=strict)
    self.load_state(state)

    return model, state

  def load_aux_state(self, state, **kwargs):
    loaded = []
    for name, obj in kwargs.items():
      ostate = state.get(name)
      if ostate is not None:
        alog.debug0(f'Loading "{name}" state')
        obj.load_state_dict(ostate)
        loaded.append(name)

    return tuple(loaded)

  @classmethod
  def load(cls, path, device=None, strict=True):
    state = ut.load_data(path, strict=strict)

    model = state['model']
    if device is not None:
      model = model.to(device)

    return model, state

  @classmethod
  def load_raw_state(cls, path):
    return ut.load_raw_data(path, map_location=torch.device('cpu'))

  @classmethod
  def model_state(cls, state):
    return state.get('model')

  @classmethod
  def load_model_state(cls, path):
    state = cls.load_raw_state(path)

    return cls.model_state(state)

  @classmethod
  def load_metrics(cls, path):
    state = cls.load_raw_state(path)

    return state.get('metrics')

  @classmethod
  def export_tb_metrics(cls, path, tb_writer):
    metrics = cls.load_metrics(path) or ()
    for m in metrics:
      cu.tb_write(tb_writer, m['name'], m['value'],
                  global_step=m['global_step'],
                  walltime=m['time'])

  def _times(self):
    return f'train={self.train_time.total}\t' \
      f'val={self.val_time.total}\t' \
      f'save={self.save_time.total}'

  def _val_loss(self, tctx):
    shuffle = not isinstance(tctx.val_data, torch.utils.data.IterableDataset)
    loader = dload.DataLoader(tctx.val_data,
                              batch_size=tctx.batch_size,
                              shuffle=shuffle,
                              num_workers=tctx.num_workers,
                              drop_last=tctx.drop_last)

    num_samples = dsu.dataset_size(tctx.val_data)
    alog.info(f'Running validation on {num_samples or "N/A"} samples')

    with torch.no_grad(), cu.Training(tctx.model, False):
      losses, val_start = [], self.val_time.start()
      for bid in pybi.BufferedIterator(loader, 4):
        x, y = bid.data
        if tctx.device is not None:
          x, y = x.to(tctx.device), y.to(tctx.device)

        _, loss = tctx.model(x, targets=y)
        losses.append(loss.item())

        if ((tctx.val_time is not None and time.time() > val_start + tctx.val_time) or
            (callable(tctx.should_stop) and tctx.should_stop())):
          alog.info(f'Interrupted validation at batch {bid.n} due to ' \
                    f'{datetime.timedelta(seconds=tctx.val_time)} required time stop')
          break

    self.val_time.track()

    return np.mean(losses) if losses else None

  def _metric_log(self, tb_writer, name, value):
    self.metrics.append(dict(name=name,
                             global_step=self.global_step,
                             time=self.train_time.seconds,
                             value=value))
    cu.tb_write(tb_writer, name, value,
                global_step=self.global_step,
                walltime=self.train_time.seconds)

  def _log_train_loss(self, loss, batch_num, step_time, tctx):
    self._metric_log(tctx.tb_writer, 'loss.train', loss)
    alog.info(f'Batch {batch_num + 1} ({self.num_samples:.1e} samples): ' \
              f'Train Loss {loss:.4f}')
    alog.info(f'Times: {self._times()}')
    alog.info(f'Perf: {tctx.batch_size / step_time:.2e} samples/sec')

  def _run_validation(self, batch_num, tctx):
    vloss = self._val_loss(tctx)
    if vloss is not None:
      self._metric_log(tctx.tb_writer, 'loss.validation', vloss)
      alog.info(f'Batch {batch_num + 1} ({self.num_samples:.1e} samples): ' \
                f'Validation Loss {vloss:.4f}')

    return vloss

  def _log_tensor_stats(self, name_fmt, stats, tctx):
    for ts in stats:
      value = {k: getattr(ts, k)
               for k in pyu.comma_split('min, max, mean, std')}
      for p, pv in zip(ts.percentiles, ts.percentile_values):
        value[f'p{int(100 * p)}'] = pv

      self._metric_log(tctx.tb_writer, name_fmt.format(ts.name), value)

  def _show_stats(self, model, tctx):
    percentiles = (0.5, 0.9, 0.95, 0.99)

    pstats = du.get_parameters_stats(model, percentiles=percentiles)
    du.show_tensors_stats(pstats, dict(value_stats=alog.DEBUG0))

    gstats = du.get_grads_stats(model, percentiles=percentiles)
    du.show_tensors_stats(gstats, dict(value_stats=alog.DEBUG0))

    self._log_tensor_stats('PARA.{}', pstats.stats, tctx)
    self._log_tensor_stats('GRAD.{}', gstats.stats, tctx)

    if current_lr := cu.get_lr(tctx.optimizer, reduce=False):
      alog.debug0(f'Current LR: {pyu.format(current_lr, ".3e")}')
      for name, lr in pyu.name_values('lr', current_lr):
        self._metric_log(tctx.tb_writer, name, lr)

  def _save_checkpoint(self, tctx):
    checkpoint = tctx.checkpoint or ('optimizer', 'scheduler', 'scaler')
    cargs = dict()
    for name in checkpoint:
      data = getattr(tctx, name, None)
      if data is not None:
        cargs[name] = data

    self.save_model(tctx.model, tctx.model_path, **cargs)

  def _step(self, tctx):
    shuffle = not isinstance(tctx.train_data, torch.utils.data.IterableDataset)
    loader = dload.DataLoader(tctx.train_data,
                              batch_size=tctx.batch_size,
                              shuffle=shuffle,
                              num_workers=tctx.num_workers,
                              drop_last=tctx.drop_last)

    num_samples = dsu.dataset_size(tctx.train_data)
    alog.info(f'Running EPOCH train on {num_samples or "N/A"} samples')

    tctx.model.train()
    tctx.optimizer.zero_grad()

    for bid in pybi.BufferedIterator(loader, tctx.accum_steps + 1):
      x, y = bid.data
      if tctx.device is not None:
        x, y = x.to(tctx.device), y.to(tctx.device)

      loss = cu.train_step(tctx.model, x, y, tctx.optimizer,
                           scaler=tctx.scaler,
                           device=tctx.device,
                           amp_dtype=tctx.amp_dtype,
                           accum_steps=tctx.accum_steps,
                           stepno=bid.n + 1,
                           grad_clip=tctx.grad_clip,
                           zero_grad=False)

      self.num_samples += len(x)
      self.global_step += 1
      if (bid.n + 1) % tctx.accum_steps == 0:
        yield pyu.make_object(stepno=bid.n, loss=loss, left=bid.left)

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
                  drop_last=True,
                  should_stop=None,
                  step_fn=None,
                  scaler=None,
                  amp_dtype=None):
    tctx = pyu.locals_capture(locals())

    wrapped_scheduler = lrw.wrap(scheduler)

    tstep, tval, tsave = [self.train_time.start()] * 3
    train_losses, val_losses, last_stepno = array.array('f'), array.array('f'), -1
    for sd in self._step(tctx):
      now = self.train_time.track()
      if now > tstep + loss_logstep:
        train_losses.append(sd.loss.item())
        self._log_train_loss(train_losses[-1],
                             sd.stepno,
                             (now - tstep) / (sd.stepno - last_stepno),
                             tctx)
        tstep, last_stepno = now, sd.stepno

      wrapped_scheduler.train_step(sd.loss)

      if model_path is not None and now > tsave + model_chkptstep:
        self.train_time.track()
        self._save_checkpoint(tctx)
        tsave = self.train_time.start()

      if now > tval + val_logstep or accum_steps > sd.left:
        self.train_time.track()
        self._show_stats(model, tctx)
        vloss = self._run_validation(sd.stepno, tctx)
        if vloss is not None:
          val_losses.append(vloss)
        tval = self.train_time.start()

      if step_fn is not None:
        step_fn(sd)

      stopped = callable(should_stop) and should_stop()
      if stopped:
        alog.info(f'Interrupted at batch {sd.stepno + 1}!')
        break

    optimizer.step()

    if not stopped:
      wrapped_scheduler.epoch_step(np.mean(val_losses) if val_losses else None)

    self.train_time.track()

    if model_path is not None:
      self._save_checkpoint(tctx)

    return train_losses, val_losses

