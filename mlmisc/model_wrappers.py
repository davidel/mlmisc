import einops
import huggingface_hub as hfh
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms.v2 as transforms
import transformers as trs

import py_misc_utils.alog as alog
import py_misc_utils.assert_checks as tas
import py_misc_utils.num_utils as pynu
import py_misc_utils.utils as pyu

from . import config as conf
from . import layer_utils as lu
from . import loss_wrappers as lsw
from . import module_builder as mbld
from . import utils as ut
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
    model_class = getattr(trs, model_class)
    model = model_class.from_pretrained(
      model_name,
      trust_remote_code=False,
      cache_dir=cache_dir)

    processor_class = getattr(trs, processor_class)
    processor = processor.from_pretrained(
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

  def forward(self, *args, **kwargs):
    with torch.no_grad():
      if args:
        model_input = self.ctx.processor(*args)
      else:
        model_input = kwargs

      y = self.ctx.model(**model_input, output_hidden_states=True)

      return y.last_hidden_state


class HugginFaceImgTune(HugginFaceModel):

  def __init__(self, model_name, model_class, loss, num_classes,
               processor_class=None,
               output_collate=None,
               dropout=None,
               act=None,
               cache_dir=None):
    processor_class = pyu.value_or(processor_class, 'AutoImageProcessor')
    output_collate = pyu.value_or(output_collate, nn.Identity())
    dropout = pyu.value_or(dropout, 0.1)
    act = pyu.value_or(act, nn.ReLU)

    super().__init__(model_name, model_class, processor_class,
                     cache_dir=cache_dir)

    alog.debug(f'Image Processor:\n{self.processor()}')

    hidden_size = max(64 * num_classes, 4 * self.config().hidden_size)

    self.loss = lsw.CatLoss(conf.create_loss(loss))
    self.output_collate = output_collate
    self.head = mbld.ModuleBuilder((self.config().hidden_size,))
    self.head.linear(hidden_size)
    self.head.add(nn.Dropout(dropout))
    self.head.add(lu.create(act))
    self.head.linear(num_classes)

  def forward(self, x, targets=None):
    with torch.no_grad():
      y = super().forward(x)
      y = self.output_collate(y)

    y = self.head(y)

    return y, self.loss(y, targets)

