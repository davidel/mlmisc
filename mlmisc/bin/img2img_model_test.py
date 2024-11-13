import argparse
import collections
import einops
import os

import matplotlib.pyplot as plt
import mlmisc.load_state_dict as mlsd
import mlmisc.trainer as mltr
import mlmisc.utils as mlut
import numpy as np
import pandas as pd
import py_misc_utils.alog as alog
import py_misc_utils.app_main as pyam
import py_misc_utils.assert_checks as tas
import py_misc_utils.break_control as pybc
import py_misc_utils.gen_fs as gfs
import py_misc_utils.module_utils as pymu
import py_misc_utils.pd_utils as pyp
import py_misc_utils.utils as pyu
import torch
import torch.nn as nn

from . import base_setup as bs
from . import dataset as ds


def load_model(args):
  module = pymu.import_module(args.model_path, modname='train_module')

  model, state = mltr.Trainer.load(args.checkpoint_path,
                                   device=args.device,
                                   strict=args.strict)

  alog.debug(f'Model Network:\n{model}')
  alog.debug(f'Model Parameters:')
  for name, param in model.named_parameters():
    alog.debug(f'  {name}\t{tuple(param.shape)}')
  alog.debug(f'Model {pyu.cname(model)} has {mlut.count_params(model):.2e} parameters')

  return model


def gen_image(img):
  img = torch.clamp(img, 0.0, 1.0)
  img = einops.rearrange(img.cpu(), 'c h w -> h w c')

  return img


def show_images(inputs, results, targets):
  for i in range(len(results)):
    iimg = gen_image(inputs[i])
    yimg = gen_image(results[i])
    timg = gen_image(targets[i])

    fig, axs = plt.subplots(1, 3, figsize=(8, 6), dpi=128)

    axs[0].set_tile('Input')
    axs[0].imshow(iimg, interpolation='bicubic')

    axs[1].set_tile('Output')
    axs[1].imshow(yimg, interpolation='bicubic')

    axs[2].set_tile('Original')
    axs[2].imshow(timg, interpolation='bicubic')

    plt.show()


def main(args):
  bs.setup(args)

  train_dataset, test_dataset = ds.create_dataset(args)

  model = load_model(args)

  with pybc.BreakControl() as bc, torch.no_grad(), mlut.Training(model, False):
    batch_size = min(args.batch_size, args.max_samples)
    loader = torch.utils.data.DataLoader(test_dataset,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         num_workers=args.num_workers)

    samples = 0
    for i, (x, y) in enumerate(loader):
      if args.device is not None:
        x, y = x.to(args.device), y.to(args.device)

      out, _ = model(x)

      show_images(x, out, y)

      samples += x.shape[0]
      if bc.hit() or samples >= args.max_samples:
        break


if __name__ == '__main__':
  # Do basic logging setup ... will be setup later once the modules are parsed.
  alog.basic_setup()

  parser = argparse.ArgumentParser(description='Image2Image Model Test',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  bs.add_parser_arguments(parser)
  ds.add_parser_arguments(parser)

  parser.add_argument('--model_path', required=True,
                      help='The path containing the model definition')
  parser.add_argument('--checkpoint_path', required=True,
                      help='The path to be used to load the model checkpoint')
  parser.add_argument('--batch_size', type=int, default=16,
                      help='The batch size to be used')
  parser.add_argument('--max_samples', type=int, default=1,
                      help='The number of samples to generate')
  parser.add_argument('--num_workers', type=int, default=max(1, os.cpu_count() // 2),
                      help='The number of workers to be used by the data loaders')
  parser.add_argument('--strict', default='true',
                      choices=tuple(mlsd.VALID_STRICTS.keys()),
                      help='Which strict mode to use when loading model state dictionaries')

  pyam.main(parser, main)

