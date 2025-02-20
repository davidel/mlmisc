import py_misc_utils.alog as alog
import py_misc_utils.utils as pyu

from .. import load_state_dict as lsd


class NoOpScheduler:

  def train_step(self, loss):
    pass

  def epoch_step(self, loss):
    pass

  def get_last_lr(self):
    pass

  def state_dict(self, *args, **kwargs):
    return dict()

  def load_state_dict(self, *args, **kwargs):
    pass


class SchedWrapper:

  NO_STATE = '_scheduler'

  def __init__(self, scheduler, **kwargs):
    self._scheduler = scheduler
    self._is_train_step = kwargs.pop('is_train_step', False)

  def train_step(self, loss):
    if self._is_train_step:
      self.step()
    else:
      train_step = getattr(self._scheduler, 'train_step', None)
      if train_step is not None:
        train_step(loss)

  def epoch_step(self, loss):
    epoch_step = getattr(self._scheduler, 'epoch_step', None)
    if epoch_step is not None:
      epoch_step(loss)
    elif not self._is_train_step:
      self.step()
      alog.debug(f'Scheduler step: lr={pyu.format(self.get_last_lr(), ".3e")}')

  def step(self):
    self._scheduler.step()

  def get_last_lr(self):
    return self._scheduler.get_last_lr()

  def state_dict(self, *args, **kwargs):
    state = self._scheduler.state_dict(*args, **kwargs)
    lsd.save_obj_state(self, state, no_state_fields=self.NO_STATE)

    return state

  def load_state_dict(self, state, *args, **kwargs):
    lsd.load_obj_state(self, state)

    self._scheduler.load_state_dict(state, *args, **kwargs)


def wrap(scheduler, **kwargs):
  if scheduler is not None:
    # No need to double-wrap if the scheduler already supports the train_step() and
    # epoch_step() APIs.
    if not isinstance(scheduler, SchedWrapper):
      return SchedWrapper(scheduler, **kwargs)
    else:
      return scheduler

  return NoOpScheduler()

