import py_misc_utils.alog as alog
import py_misc_utils.assert_checks as tas
import py_misc_utils.module_utils as pymu
import py_misc_utils.utils as pyu
import torch
import torch.nn as nn
import torchvision

from . import config as conf
from . import layer_utils as lu
from . import loss_wrappers as lsw
from . import module_builder as mbld
from . import web_module as wmod


class TorchVision(nn.Module):

  def __init__(self, name, loss, *args, **kwargs):
    super().__init__()
    self.loss = lsw.CatLoss(conf.create_loss(loss))
    self.mod = torchvision.models.get_model(name, *args, **kwargs)

  def forward(self, x, targets=None):
    y = self.mod(x)

    return y, self.loss(y, targets)


class Web(nn.Module):

  def __init__(self, repo, module, ctor, loss, *args,
               cache_dir=None,
               commit=None,
               force_clone=None,
               **kwargs):
    super().__init__()
    self.loss = lsw.CatLoss(conf.create_loss(loss))
    self.mod = wmod.WebModule(repo, module, ctor,
                              cache_dir=cache_dir,
                              commit=commit,
                              force_clone=force_clone,
                              mod_args=args,
                              mod_kwargs=kwargs)

  def forward(self, x, targets=None):
    y = self.mod(x)

    return y, self.loss(y, targets)


class HugginFaceModel(nn.Module):

  def __init__(self, model_name, model_class, processor_class,
               cache_dir=None):
    cache_dir = pyu.cache_dir(path=cache_dir)

    mclass, = pymu.import_module_names(model_class)
    model = mclass.from_pretrained(
      model_name,
      trust_remote_code=False,
      cache_dir=cache_dir)

    pclass, = pymu.import_module_names(processor_class)
    processor = pclass.from_pretrained(
      model_name,
      use_fast=True,
      trust_remote_code=False,
      cache_dir=cache_dir)

    alog.debug(f'HF Model:\n{model}')
    alog.debug(f'Model Config:\n{model.config}')

    super().__init__()
    # Storing model inside an object prevents nn.Module machinery to save its
    # weights when checkpointing, which is what we want as they are frozen.
    self.ctx = pyu.make_object(model=model, processor=processor)

  def model(self):
    return self.ctx.model

  def config(self):
    return self.ctx.model.config

  def processor(self):
    return self.ctx.processor

  def to(self, *args, **kwargs):
    res = super().to(*args, **kwargs)
    res.ctx.model = res.ctx.model.to(*args, **kwargs)

    return res

  def forward(self, *args, processor_kwargs=None, **kwargs):
    with torch.no_grad():
      if processor_kwargs is not None:
        model_kwargs = self.ctx.processor(*args, **processor_kwargs)
        model_kwargs.update(kwargs)
        model_args = ()
      else:
        model_args, model_kwargs = args, kwargs

      y = self.ctx.model(*model_args, **model_kwargs, output_hidden_states=True)

      return y.last_hidden_state


class HugginFaceImgTune(HugginFaceModel):

  def __init__(self, model_name, model_class, head, loss,
               processor_class=None,
               cache_dir=None):
    processor_class = pyu.value_or(processor_class, 'transformers.AutoImageProcessor')

    super().__init__(model_name, model_class, processor_class,
                     cache_dir=cache_dir)

    alog.debug(f'Image Processor:\n{self.processor()}')

    self.loss = lsw.CatLoss(conf.create_loss(loss))
    self.head = conf.create_model(head, self.config())

  def forward(self, x, targets=None):
    with torch.no_grad():
      y = super().forward(x, processor_kwargs={})

    y = self.head(y)

    return y, self.loss(y, targets)


class HugginFaceSeqTune(HugginFaceModel):

  def __init__(self, model_name, model_class, head, loss,
               processor_class=None,
               cache_dir=None):
    processor_class = pyu.value_or(processor_class, 'transformers.AutoTokenizer')

    super().__init__(model_name, model_class, processor_class,
                     cache_dir=cache_dir)

    self.loss = lsw.SeqLoss(conf.create_loss(loss))
    self.head = conf.create_model(head, self.config(), self.processor())

  def forward(self, x, targets=None):
    with torch.no_grad():
      y = super().forward(x)

    y = self.head(y)

    return y, self.loss(y, targets)

