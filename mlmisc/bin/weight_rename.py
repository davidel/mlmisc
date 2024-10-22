import argparse
import re
import yaml

import mlmisc.trainer as mltr
import mlmisc.utils as mlut
import py_misc_utils.alog as alog
import torch


def load_data(path, map_location):
  alog.debug(f'Loading {args.input} checkpoint')
  state = mltr.Trainer.load_raw_state(args.input)

  return state, mltr.Trainer.model_state(state)


def analyze(args):
  _, from_data = load_data(args.from_path, args.map_location)
  _, to_data = load_data(args.to_path, args.map_location)


def dump(args):
  _, model_data = load_data(args.input, args.map_location)

  names = sorted(model_data.keys())

  od, rd = dict(), dict()
  for i, name in enumerate(names):
    param = model_data[name]
    od[str(i)] = dict(name=name, shape=str(tuple(param.shape)))
    rd[str(i)] = name

  with open(args.dump_file, mode='wt') as df:
    yaml.dump(dict(orig=od, replace=rd), df, default_flow_style=None)


def replace(args):
  data, model_data = load_data(args.input, args.map_location)

  repl = []
  for k in model_data.keys():
    for r in args.replace:
      mx, rx = r.split(',')

      nk = re.sub(mx, rx, k)
      if nk != k:
        repl.append((k, nk))

  for k, nk in repl:
    alog.info(f'Renaming "{k}" to "{nk}"')
    model_data[nk] = model_data[k]
    model_data.pop(k)

  if repl:
    wpath = args.input + '.replace' if args.output is None else args.output
    alog.debug(f'Saving updated checkpoint to {wpath}')
    torch.save(data, wpath)


if __name__ == '__main__':
  parser = argparse.ArgumentParser('Manipulate parameter names from PyTorch checkpoints')
  parser.add_argument('--map_location')

  subparsers = parser.add_subparsers(required=True,
                                     help='Command help')

  replace_parser = subparsers.add_parser('replace', help='Replaces model parameter names')
  replace_parser.add_argument('--input', required=True)
  replace_parser.add_argument('--replace', nargs='+')
  replace_parser.add_argument('--output')
  replace_parser.set_defaults(cmd_fn=replace)

  dump_parser = subparsers.add_parser('dump', help='Dumps model parameter names')
  dump_parser.add_argument('--input', required=True)
  dump_parser.add_argument('--dump_file', required=True)
  dump_parser.set_defaults(cmd_fn=dump)

  analyze_parser = subparsers.add_parser('analyze', help='Analyzes model parameter names')
  analyze_parser.add_argument('--from_path', required=True)
  analyze_parser.add_argument('--to_path', required=True)
  analyze_parser.set_defaults(cmd_fn=analyze)

  args = parser.parse_args()
  args.cmd_fn(args)

