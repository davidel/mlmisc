import argparse
import re

import mlmisc.utils as mlut
import py_misc_utils.alog as alog
import torch


def main(args):
  alog.debug(f'Loading {args.input} checkpoint')
  data = mlut.torch_load(args.input, map_location=args.map_location)

  # Handle ut.save_data() packages ...
  model_data = data.get('model', data)

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
  parser = argparse.ArgumentParser('Change parameter names from PyTorch checkpoints')
  parser.add_argument('--input', required=True)
  parser.add_argument('--replace', nargs='+')
  parser.add_argument('--output')
  parser.add_argument('--map_location')

  args = parser.parse_args()

  main(args)

