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


def get_classes(dataset):
  classes = dataset.extra_arg('classes')
  if classes:
    classes = tuple(c[0] for c in classes) if isinstance(classes[0], (list, tuple)) else classes
    alog.info(f'Dataset Classes:')
    for i, cls in enumerate(classes):
      alog.info(f'  {i:04d} = "{cls}"')

  return classes


def class_name(idx, classes):
  idx = mlut.item(idx)

  return classes[idx] if classes else f'CLS{idx:04d}'


def show_balance(dsname, dataset, classes):
  classes_counts = collections.defaultdict(int)
  for i in range(len(dataset)):
    x, y = dataset[i]
    classes_counts[class_name(y, classes)] += 1

  counts = [f'{k}={c}' for k, c in classes_counts.items()]
  alog.info(f'{dsname} Balance: {", ".join(counts)}')


def report_mismatches(args, x, targets, predicted, mismatch_indices, classes,
                      class_misses, num_processed):
  for u in mismatch_indices:
    tclass = class_name(targets[u], classes)
    pclass = class_name(predicted[u], classes)
    class_misses[targets[u]][predicted[u]] += 1

    if args.display_mismatches or args.report_path is not None:
      imgdata = torch.clamp(x[u], 0.0, 1.0)
      imgdata = einops.rearrange(imgdata.cpu(), 'c h w -> h w c')

      plt.figure(figsize=(8, 6), dpi=128)

      plt.title(f'Correct="{tclass}" Predicted="{pclass}"')
      plt.imshow(imgdata, interpolation='bicubic')

      if args.report_path is not None:
        spath = os.path.join(args.report_path, tclass, pclass)
        gfs.makedirs(spath, exist_ok=True)

        imgpath = os.path.join(spath, f'dsn_{num_processed + u}.jpg')
        with gfs.open(imgpath, mode='wb') as imgfd:
          plt.savefig(imgfd)

      if args.display_mismatches:
        plt.show()


def emit_class_misses(args, class_misses, classes, max_class):
  if args.report_path is not None:
    mdata = [[class_name(i, classes) for i in range(max_class)]]
    mdata += [np.zeros(max_class, dtype=int) for _ in range(max_class)]
    for ti, pd in class_misses.items():
      for pi, count in pd.items():
        mdata[ti][pi + 1] = count

    columns = ['TARGET_CLASS'] + [class_name(i, classes) for i in range(max_class)]

    df = pd.DataFrame(data=mdata, columns=columns)

    gfs.makedirs(args.report_path, exist_ok=True)
    pyp.save_dataframe(df, os.path.join(args.report_path, 'class_misses.csv'))


def main(args):
  bs.setup(args)

  train_dataset, test_dataset = ds.create_dataset(args)

  classes = get_classes(test_dataset)
  show_balance('Test', test_dataset, classes)

  model = load_model(args)

  with pybc.BreakControl() as bc, torch.no_grad(), mlut.Training(model, False):
    loader = torch.utils.data.DataLoader(test_dataset,
                                         batch_size=args.batch_size,
                                         shuffle=False,
                                         num_workers=args.num_workers)

    class_misses = collections.defaultdict(lambda: collections.defaultdict(int))

    num_processed, num_correct, max_class = 0, 0, None
    for i, (x, y) in enumerate(loader):
      if args.device is not None:
        x, y = x.to(args.device), y.to(args.device)

      out, _ = model(x)

      max_class = out.shape[-1]

      _, predicted = torch.max(out, dim=-1)
      targets, predicted = y.flatten(), predicted.flatten()
      match_mask = predicted == targets

      mismatch_indices = torch.nonzero(~match_mask).flatten().tolist()
      report_mismatches(args, x, targets.tolist(), predicted.tolist(), mismatch_indices,
                        classes, class_misses, num_processed)

      num_correct += match_mask.sum().item()
      num_processed += len(match_mask)
      alog.info(f'Precision: {100 * num_correct / num_processed:.2f}%')

      if bc.hit():
        break

    emit_class_misses(args, class_misses, classes, max_class)


if __name__ == '__main__':
  # Do basic logging setup ... will be setup later once the modules are parsed.
  alog.basic_setup()

  parser = argparse.ArgumentParser(description='Image Model Test',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  bs.add_parser_arguments(parser)
  ds.add_parser_arguments(parser)

  parser.add_argument('--model_path', required=True,
                      help='The path containing the model definition')
  parser.add_argument('--checkpoint_path', required=True,
                      help='The path to be used to load the model checkpoint')
  parser.add_argument('--batch_size', type=int, default=32,
                      help='The batch size to be used')
  parser.add_argument('--num_workers', type=int, default=max(1, os.cpu_count() // 2),
                      help='The number of workers to be used by the data loaders')
  parser.add_argument('--strict', default='true',
                      choices=tuple(mlsd.VALID_STRICTS.keys()),
                      help='Which strict mode to use when loading model state dictionaries')
  parser.add_argument('--report_path',
                      help='The path where the report should be generated')
  parser.add_argument('--display_mismatches', action=argparse.BooleanOptionalAction,
                      default=False,
                      help='Whether to display the images which failed detection')

  pyam.main(parser, main)

