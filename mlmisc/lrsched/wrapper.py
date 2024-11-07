import py_misc_utils.alog as alog
import py_misc_utils.utils as pyu


class NoOp:

  def train_step(self, loss):
    pass

  def epoch_step(self, loss):
    pass

  def get_last_lr(self):
    pass


class Wrapper:

  def __init__(self, scheduler, **kwargs):
    self.scheduler = scheduler
    self.is_train_step = kwargs.pop('is_train_step', False)

  def train_step(self, loss):
    if self.is_train_step:
      self.scheduler.step()
    else:
      train_step = getattr(self.scheduler, 'train_step', None)
      if train_step is not None:
        return train_step(loss)

  def epoch_step(self, loss):
    epoch_step = getattr(self.scheduler, 'epoch_step', None)
    if epoch_step is not None:
      return epoch_step(loss)
    elif not self.is_train_step:
      self.scheduler.step()
      alog.debug(f'Scheduler step: lr={pyu.format(self.scheduler.get_last_lr(), ".3e")}')

  def get_last_lr(self):
    return self.scheduler.get_last_lr()


def wrap(scheduler, **kwargs):
  return Wrapper(scheduler, **kwargs) if scheduler is not None else NoOp()

