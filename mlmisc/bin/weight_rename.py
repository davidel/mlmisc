import argparse
import re

import py_misc_utils.alog as alog
import torch


def main(args):
  alog.debug(f'Loading {args.input} checkpoint')
  data = torch.load(args.input, weights_only=False, map_location=args.map_location)

  repl = []
  for k in data.keys():
    for r in args.replace:
      mx, rx = r.split(',')

      nk = re.sub(mx, rx, k)
      if nk != k:
        repl.append((k, nk))

  for k, nk in repl:
    alog.info(f'Renaming "{k}" to "{nk}"')
    data[nk] = data[k]
    data.pop(k)

  if repl:
    wpath = args.input + '.replace' if args.output is None else args.output
    alog.debug(f'Saving updated checkpoint to {wpath}')
    torch.save(data, wpath)


if __name__ == '__main__':
  parser = argparse.ArgumentParser('Change parameter names from Pytorch checkpoints')
  parser.add_argument('--input', required=True)
  parser.add_argument('--replace', nargs='+')
  parser.add_argument('--output')
  parser.add_argument('--map_location')

  args = parser.parse_args()

  main(args)

