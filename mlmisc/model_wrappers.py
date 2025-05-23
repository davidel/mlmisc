import py_misc_utils.alog as alog
import py_misc_utils.assert_checks as tas
import py_misc_utils.context_managers as pycm
import py_misc_utils.gfs as gfs
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


class ModelHolder(nn.Module):

  def __init__(self, model, frozen=True, **kwargs):
    super().__init__()
    self._frozen = frozen
    # Storing model inside an object prevents nn.Module machinery to save its
    # weights when checkpointing, which is what we want when they are frozen.
    self._ctx = pyu.make_object(model=model, **kwargs)
    if not frozen:
      self.model = model

  def to(self, *args, **kwargs):
    super().to(*args, **kwargs)
    if self._frozen:
      self._ctx.model.to(*args, **kwargs)

    return self

  def model_ctx(self):
    return pycm.Pack(torch.set_grad_enabled(not self._frozen),
                     wrap_obj=self._ctx.model)


class TorchVision(ModelHolder):

  def __init__(self, name, loss, *args,
               frozen=None,
               **kwargs):
    model = torchvision.models.get_model(name, *args, **kwargs)

    super().__init__(model, frozen=frozen)
    self.loss = lsw.CatLoss(conf.create_loss(loss))

  def forward(self, x, targets=None):
    with self.model_ctx() as model:
      y = model(x)

    return y, self.loss(y, targets)


class Web(ModelHolder):

  def __init__(self, repo, module, ctor, loss, *args,
               frozen=None,
               commit=None,
               force_clone=None,
               **kwargs):
    model = wmod.WebModule(repo, module, ctor,
                           commit=commit,
                           force_clone=force_clone,
                           mod_args=args,
                           mod_kwargs=kwargs)

    super().__init__(model, frozen=frozen)
    self.loss = lsw.CatLoss(conf.create_loss(loss))

  def forward(self, x, targets=None):
    with self.model_ctx() as model:
      y = model(x)

    return y, self.loss(y, targets)


class HugginFaceModel(ModelHolder):

  def __init__(self, model_name, model_class, processor_class,
               frozen=None):
    cache_dir = gfs.cache_dir()

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

    super().__init__(model, frozen=frozen, processor=processor)

  def config(self):
    return self._ctx.model.config

  def processor(self):
    return self._ctx.processor

  def forward(self, *args, processor_kwargs=None, **kwargs):
    with self.model_ctx() as model:
      if processor_kwargs is not None:
        model_kwargs = self._ctx.processor(*args, **processor_kwargs)
        model_kwargs.update(kwargs)
        model_args = ()
      else:
        model_args, model_kwargs = args, kwargs

      y = model(*model_args, **model_kwargs, output_hidden_states=True)

      return y.last_hidden_state


class HugginFaceImgTune(HugginFaceModel):

  def __init__(self, model_name, model_class, head, loss,
               processor_class='transformers.AutoImageProcessor',
               frozen=None):
    super().__init__(model_name, model_class, processor_class,
                     frozen=frozen)

    alog.debug(f'Image Processor:\n{self.processor()}')

    self.loss = lsw.CatLoss(conf.create_loss(loss))
    self.head = conf.create_model(head,
                                  config=self.config(),
                                  processor=self.processor())

  def forward(self, x, targets=None):
    y = super().forward(x, processor_kwargs={})
    y = self.head(y)

    return y, self.loss(y, targets)


class HugginFaceSeqTune(HugginFaceModel):

  def __init__(self, model_name, model_class, context_size, head, loss,
               processor_class='transformers.AutoTokenizer',
               frozen=None):
    super().__init__(model_name, model_class, processor_class,
                     frozen=frozen)

    self.loss = lsw.SeqLoss(conf.create_loss(loss))
    self.head = conf.create_model(head,
                                  config=self.config(),
                                  tokenizer=self.processor(),
                                  context_size=context_size)

  def forward(self, x, targets=None):
    y = super().forward(x)
    y = self.head(y)

    return y, self.loss(y, targets)

