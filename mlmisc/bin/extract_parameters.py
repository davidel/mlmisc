import argparse
import os
import re

import py_misc_utils.alog as alog
import py_misc_utils.app_main as pyam
import py_misc_utils.utils as pyu
import torch

from .. import trainer as tr


def _main(args):
  state = tr.Trainer.load_model_state(args.checkpoint_path)

  for name, param in state.items():
    if isinstance(param, torch.Tensor):
      alog.debug(f'Found parameter "{name}" with shape {tuple(param.shape)}')
      if not args.match or re.match(args.match, name):
        data = getattr(param, 'data', param)
        path = os.path.join(args.output_path, f'{name}.pt')

        alog.info(f'Saving parameter "{name}" to {path} ...')
        torch.save(data, path)


if __name__ == '__main__':
  # Do basic logging setup ... will be setup later once the modules are parsed.
  alog.basic_setup()

  parser = argparse.ArgumentParser(description='Extracts metrics from model checkpoints',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--checkpoint_path', required=True,
                      help='The path to be used to load the model checkpoint which contains the metrics')
  parser.add_argument('--output_path', default='.',
                      help='The destination path where the parameters files (in torch.save() ' \
                      f'format) should be created')
  parser.add_argument('--match',
                      help='The regular expression used to select the parameters to be extracted')

  pyam.main(parser, _main)

