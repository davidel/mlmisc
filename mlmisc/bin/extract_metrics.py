import argparse
import os
import shutil

import py_misc_utils.alog as alog
import py_misc_utils.app_main as pyam
import py_misc_utils.utils as pyu

from .. import trainer as tr
from .. import utils as ut


def _main(args):
  if os.path.exists(args.tb_path):
    if args.tb_overwrite:
      shutil.rmtree(args.tb_path, ignore_errors=True)
    else:
      alog.xraise(RuntimeError, f'Tensorboard ouput folder already exists: {args.tb_path}')

  tb_writer = ut.create_tb_writer(args.tb_path)

  tr.Trainer.export_tb_metrics(args.checkpoint_path, tb_writer)

  tb_writer.flush()
  tb_writer.close()


if __name__ == '__main__':
  # Do basic logging setup ... will be setup later once the modules are parsed.
  alog.basic_setup()

  parser = argparse.ArgumentParser(description='Extracts metrics from model checkpoints',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--checkpoint_path', required=True,
                      help='The path to be used to load the model checkpoint which contains the metrics')
  parser.add_argument('--tb_path', required=True,
                      help='The Tensorboard logdir path where the metrics should be exported to')
  parser.add_argument('--tb_overwrite', action=argparse.BooleanOptionalAction, default=False,
                      help='Whether an existing --tb_path folder should be overwritten')

  pyam.main(parser, _main)

