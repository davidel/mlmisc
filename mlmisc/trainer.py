import array
import datetime
import time

from mlmisc import utils as mlu
import numpy as np
from py_misc_utils import alog
import torch


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
    mlu.save_data(path, model=model, **kwargs, **state)
    self._save_time.track()

  def load_model(self, model, path, device=None):
    state = mlu.load_data(path, model=model)
    self._load_state(state)

    return model.to(device) if device is not None else model, state

  def load_aux_state(self, state, **kwargs):
    for name, obj in kwargs.items():
      ostate = state.get(name)
      if ostate is not None:
        obj.load_state_dict(ostate)

  def _times(self):
    return f'train={self._train_time.total}\t' \
      f'val={self._val_time.total}\t' \
      f'save={self._save_time.total}'

  def _val_loss(self, model, val_data, val_time=None, device=None, batch_size=None):
    loader = torch.utils.data.DataLoader(val_data,
                                         batch_size=batch_size or 1,
                                         shuffle=True)

    alog.info(f'Running validation on {len(loader)} batches')

    model.eval()
    with torch.no_grad():
      losses, val_start = [], self._val_time.start()
      for i, (x, y) in enumerate(loader):
        if device is not None:
          x, y = x.to(device), y.to(device)

        _, loss = model(x, targets=y)
        losses.append(loss.item())

        if val_time is not None and time.time() > val_start + val_time:
          alog.info(f'Validation run on {i} of {len(loader)} batches due to ' \
                    f'{datetime.timedelta(seconds=val_time)} required time stop')
          break

    model.train()

    self._val_time.track()

    return np.mean(losses)

  def train_epoch(self, model, optimizer, train_data, val_data, batch_size,
                  device=None,
                  scheduler=None,
                  val_time=None,
                  loss_logstep=60,
                  val_logstep=900,
                  model_chkptstep=600,
                  model_path=None,
                  tb_writer=None):
    loader = torch.utils.data.DataLoader(train_data,
                                         batch_size=batch_size,
                                         shuffle=True)

    num_batches = len(loader)
    alog.info(f'Running EPOCH train on {num_batches} batches')

    tstep, tval, tsave = [self._train_time.start()] * 3

    model.train()

    train_losses, val_losses = array.array('f'), array.array('f')
    for i, (x, y) in enumerate(loader):
      if device is not None:
        x, y = x.to(device), y.to(device)

      _, loss = model(x, targets=y)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      self._num_samples += batch_size

      now = self._train_time.track()
      if now > tstep + loss_logstep:
        train_losses.append(loss.item())
        if tb_writer is not None:
          tb_writer.add_scalar('Train Loss', train_losses[-1],
                               walltime=self._train_time.seconds)

        epoch = 100 * self._num_samples / (num_batches * batch_size)
        alog.info(f'Batch {i + 1}/{num_batches} (epoch={epoch:.1f}%): ' \
                  f'Train Loss {train_losses[-1]:.4f}')
        alog.info(f'Times: {self._times()}')
        tstep = now

      if model_path is not None and now > tsave + model_chkptstep:
        self._train_time.track()
        self.save_model(model, model_path,
                        optimizer=optimizer,
                        scheduler=scheduler)
        tsave = self._train_time.start()

      if now > tval + val_logstep:
        self._train_time.track()
        vloss = self._val_loss(model, val_data,
                               val_time=val_time,
                               batch_size=batch_size,
                               device=device)
        val_losses.append(vloss)
        if tb_writer is not None:
          tb_writer.add_scalar('Validation Loss', vloss, walltime=self._train_time.seconds)
        alog.info(f'Validation Loss {vloss:.4f}')
        tval = self._train_time.start()

    if scheduler is not None:
      scheduler.step(np.mean(val_losses))

    self._train_time.track()

    return train_losses, val_losses

