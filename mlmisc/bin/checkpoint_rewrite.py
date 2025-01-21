import argparse
import sys

import py_misc_utils.alog as alog
import py_misc_utils.app_main as pyam
import py_misc_utils.file_overwrite as pyfow
import py_misc_utils.gfs as gfs
import py_misc_utils.remap_pickle as pyrp
import py_misc_utils.utils as pyu
import torch


def main(args):
  remaps = None
  if args.remaps:
    remaps = dict()
    for remap in args.remaps:
      cfrom, cto = pyu.comma_split(remap)
      remaps[cfrom] = cto

  safe_refs = args.safe_refs if args.safe_refs else None

  with gfs.open(args.input, mode='rb') as fd:
    data = torch.load(fd,
                      pickle_module=pyrp,
                      remaps=remaps,
                      safe_refs=safe_refs)

  output_path = args.output or args.input
  with pyfow.FileOverwrite(output_path, mode='wb') as fd:
    torch.save(data, fd)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description='Pytorch Checkpoint Rewrite Tools',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('--input', required=True,
                      help='The path to be used to load the checkpoint')
  parser.add_argument('--output',
                      help='The path to be used to store the rewritten checkpoint')
  parser.add_argument('--remaps', nargs='+',
                      help='The comma-separated ("FROM,TO" with FROM supporting regex) remap strings')
  parser.add_argument('--safe_refs', nargs='+',
                      help='The safe references regular expressions to validate loading')

  pyam.main(parser, main)

