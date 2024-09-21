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
from . import module_builder as mbld
from . import utils as ut
from . import web_module as wmod


class TorchVision(nn.Module):

  def __init__(self, name, loss, *args, **kwargs):
    super().__init__()
    self.loss = conf.create_loss(loss)
    self.mod = torchvision.models.get_model(name, *args, **kwargs)

  def forward(self, x, targets=None):
    y = self.mod(x)

    return y, ut.compute_loss(self.loss, y, targets)


class Web(nn.Module):

  def __init__(self, repo, module, ctor, loss, *args,
               cache_dir=None,
               commit=None,
               force_clone=None,
               **kwargs):
    super().__init__()
    self.loss = conf.create_loss(loss)
    self.mod = wmod.WebModule(repo, module, ctor,
                              cache_dir=cache_dir,
                              commit=commit,
                              force_clone=force_clone,
                              mod_args=args,
                              mod_kwargs=kwargs)

  def forward(self, x, targets=None):
    y = self.mod(x)

    return y, ut.compute_loss(self.loss, y, targets)


class HugginFaceImgTune(nn.Module):

  def __init__(self, model_name, model_class, loss, num_classes,
               output_collate=None,
               cache_dir=None):
    mclass = getattr(trs, model_class)
    model = mclass.from_pretrained(
      model_name,
      trust_remote_code=False,
      cache_dir=cache_dir)

    image_processor = trs.AutoImageProcessor.from_pretrained(
      model_name,
      use_fast=True,
      trust_remote_code=False,
      cache_dir=cache_dir)

    alog.info(f'HF Model:\n{model}')
    alog.info(f'Model Config:\n{model.config}')
    alog.info(f'Image Processor:\n{image_processor}')

    hidden_size = max(64 * num_classes, 4 * model.config.hidden_size)

    super().__init__()
    self.loss = conf.create_loss(loss)
    self.output_collate = output_collate
    self.ctx = pyu.make_object(model=model, image_processor=image_processor)
    self.head = mbld.ModuleBuilder((self.ctx.model.config.hidden_size,))
    self.head.linear(hidden_size)
    self.head.add(nn.Dropout(0.2))
    self.head.add(nn.ReLU(inplace=True))
    self.head.linear(num_classes)

  def to(self, *args, **kwargs):
    res = super().to(*args, **kwargs)
    res.ctx.model = res.ctx.model.to(*args, **kwargs)

    return res

  def forward(self, x, targets=None):
    with torch.no_grad():
      px = self.ctx.image_processor(x)
      vouts = self.ctx.model(**px, output_hidden_states=True)

      if self.output_collate == 'first':
        y = vouts.last_hidden_state[:, 0, :]
      elif self.output_collate == 'last':
        y = vouts.last_hidden_state[:, -1, :]
      elif self.output_collate == 'mean':
        y = einops.reduce(vouts.last_hidden_state, 'b n h -> b h', 'mean')
      else:
        y = vouts.last_hidden_state[:, int(self.output_collate), :]

    y = self.head(y)

    return y, ut.compute_loss(self.loss, y, targets)

