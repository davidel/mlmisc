import argparse
import os

import torch

import py_misc_utils.alog as alog
import py_misc_utils.app_main as pyam
import py_misc_utils.assert_checks as tas
import py_misc_utils.gfs as pygfs
import py_misc_utils.module_utils as pymu
import py_misc_utils.utils as pyu

from .. import core_utils as cu
from .. import sequence_utils as sequ
from .. import tokenizers as tkz
from .. import utils as ut

from . import base_setup as bs


def _load_model(args):
  module = pymu.import_module(args.model_path, modname='train_module')

  state = ut.load_data(args.checkpoint_path)

  model = state['model']
  model = model.to(args.device)

  return model


def _load_tokenizer(args):
  if args.tokenizer_config is not None:
    tokenizer = tkz.from_config(args.tokenizer_config)
  elif args.tokenizer_name is None:
    tokenizer = tkz.load_tokenizer(args.tokenizer_path)
  else:
    tokenizer = tkz.from_pretrained(args.tokenizer_path, args.tokenizer_name)

  return tokenizer


def _eval_model(net):
  def forward(*args, **kwargs):
    # Model returns (output, loss) tuple. We select the output.
    result = net(*args, **kwargs)

    return result[0]

  return forward


def _generate(args, model, tokenizer):
  with torch.no_grad(), cu.Training(model, False):
    iseq = tokenizer.encode(args.input_sequence)
    iseq_tensor = torch.tensor(iseq, dtype=torch.long, device=args.device)

    gids = sequ.generate(_eval_model(model),
                         iseq_tensor,
                         args.context_size,
                         args.num_steps,
                         args.pad_mode,
                         args.pad_value,
                         temperature=args.temperature,
                         sample=True,
                         top_k=args.top_k)

    gentext = tokenizer.decode(gids.tolist())

    with pygfs.std_open(args.output_file, mode='w') as ofd:
      ofd.write(gentext)


def _main(args):
  bs.setup(args)

  model = _load_model(args)
  tokenizer = _load_tokenizer(args)
  _generate(args, model, tokenizer)


if __name__ == '__main__':
  # Do basic logging setup ... will be setup later once the modules are parsed.
  alog.basic_setup()

  parser = argparse.ArgumentParser(description='Text Generator',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  bs.add_parser_arguments(parser)

  parser.add_argument('--model_path', required=True,
                      help='The path containing the model definition')
  parser.add_argument('--checkpoint_path', required=True,
                      help='The path to be used to load the model checkpoint')
  parser.add_argument('--tokenizer_path',
                      help='The path containing the tokenizer protobuf file, or the module '\
                      f'path for a pre-trained one')
  parser.add_argument('--tokenizer_name',
                      help='The tokenizer name in case a pre-trained one needs to be loaded')
  parser.add_argument('--tokenizer_config',
                      help='The tokenizer configuration to be used')
  parser.add_argument('--input_sequence', required=True,
                      help='The path to be used to store the model checkpoint')
  parser.add_argument('--context_size', type=int, required=True,
                      help='The context size for the model')
  parser.add_argument('--num_steps', type=int, default=1000,
                      help='The number of extra tokens to be generated')
  parser.add_argument('--pad_mode', default='none',
                      choices=('none', 'front', 'back'),
                      help='The pad mode for the model input sequence')
  parser.add_argument('--pad_value', type=int, default=0,
                      help='The pad value')
  parser.add_argument('--temperature', type=float,
                      help='The temperature value')
  parser.add_argument('--top_k', type=int,
                      help='The top number of tokens to restrict the selection to')
  parser.add_argument('--output_file', default='STDOUT',
                      help='The file where the output should be written (supports STDOUT, STDERR)')

  pyam.main(parser, _main)

