import argparse
import os
import re
import yaml

import mlmisc.trainer as mltr
import mlmisc.utils as mlut
import py_misc_utils.alog as alog
import py_misc_utils.utils as pyu
import torch


def load_data(path, map_location):
  alog.debug(f'Loading {args.input} checkpoint')
  state = mltr.Trainer.load_raw_state(args.input)

  return state, mltr.Trainer.model_state(state)


def rewrite_data(args, data):
  wpath = args.input + '.replace' if args.output is None else args.output
  alog.debug(f'Saving updated checkpoint to {wpath}')
  torch.save(data, wpath)


def trans_data(param, trans):
  if os.path.isfile(trans):
    with open(trans, mode='rt') as cfd:
      code = cfd.read()

    transfn, = pyu.compile(code, 'transfn')

    return transfn(param)
  else:
    return eval(trans, dict(data=param))


def analyze(args):
  _, from_data = load_data(args.from_path, args.map_location)
  _, to_data = load_data(args.to_path, args.map_location)


def dump(args):
  _, model_data = load_data(args.input, args.map_location)

  names = sorted(model_data.keys())

  orig, replace = dict(), dict()
  for i, name in enumerate(names):
    param = model_data[name]
    if isinstance(param, torch.Tensor):
      pkey = f'P{i}'
      orig[pkey] = dict(name=name, shape=str(tuple(param.shape)))
      replace[pkey] = dict(name=name)
    else:
      alog.debug(f'Key "{name}" is not a tensor: {type(param)}')

  pyu.write_config(dict(orig=orig, replace=replace), args.dump_file)


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
    rewrite_data(args, data)


def dreplace(args):
  data, model_data = load_data(args.input, args.map_location)

  repl_config = pyu.load_config(cfg_file=args.dump_file)

  orig, replace = pyu.mget(repl_config, 'orig, replace')
  changed = 0
  for pid, pdata in orig.items():
    rdata = replace.get(pid)
    if rdata is not None:
      param = model_data[pdata['name']]
      trans = rdata.get('trans')
      if trans is not None:
        alog.info(f'Transforming "{pdata["name"]}": {trans}')
        param = trans_data(param, trans)

      if rdata['name'] != pdata['name']:
        alog.info(f'Renaming "{pdata["name"]}" with shape {pdata["shape"]} to "{rdata["name"]}"')
        model_data[rdata['name']] = param
        model_data.pop(pdata['name'])
        changed += 1
    else:
      alog.info(f'Dropping "{pdata["name"]}" with shape {pdata["shape"]}')
      model_data.pop(pdata['name'])
      changed += 1

  if changed:
    rewrite_data(args, data)


if __name__ == '__main__':
  parser = argparse.ArgumentParser('Manipulate parameter names from PyTorch checkpoints')
  parser.add_argument('--map_location')
  parser.add_argument('--log_level', default='DEBUG')

  subparsers = parser.add_subparsers(required=True,
                                     help='Command help')

  replace_parser = subparsers.add_parser(
    'replace',
    help='Replaces model parameter names')
  replace_parser.add_argument('--input', required=True)
  replace_parser.add_argument('--replace', nargs='+')
  replace_parser.add_argument('--output')
  replace_parser.set_defaults(cmd_fn=replace)

  dump_parser = subparsers.add_parser(
    'dump',
    help='Dumps model parameter names')
  dump_parser.add_argument('--input', required=True)
  dump_parser.add_argument('--dump_file', required=True)
  dump_parser.set_defaults(cmd_fn=dump)

  dreplace_parser = subparsers.add_parser(
    'dreplace',
    help='Replaces model parameter names using rewritten dump file')
  dreplace_parser.add_argument('--input', required=True)
  dreplace_parser.add_argument('--dump_file', required=True)
  dreplace_parser.add_argument('--output')
  dreplace_parser.set_defaults(cmd_fn=dreplace)

  analyze_parser = subparsers.add_parser(
    'analyze',
    help='Analyzes model parameter names')
  analyze_parser.add_argument('--from_path', required=True)
  analyze_parser.add_argument('--to_path', required=True)
  analyze_parser.set_defaults(cmd_fn=analyze)

  args = parser.parse_args()

  alog.basic_setup(log_level=args.log_level)

  args.cmd_fn(args)

