import argparse
import collections
import os
import yaml

import py_misc_utils.alog as alog
import py_misc_utils.utils as pyu
import torch

from .. import core_utils as cu


def expand(data, dest_folder, ctx):
  if isinstance(data, collections.abc.Mapping):
    xdata = dict()
    for name, value in data.items():
      xdata[name] = expand(value, dest_folder, ctx)

    return xdata
  else:
    dpath = os.path.join(dest_folder, f'{ctx.seqidx}.pt')
    ctx.seqidx += 1

    torch.save(data, dpath)

    return os.path.basename(dpath)


def extract_data(path, dest_folder):
  if os.path.exists(dest_folder):
    alog.xraise(RuntimeError, f'Destination folder already exists: {dest_folder}')

  alog.debug(f'Extracting {path} checkpoint to {dest_folder}')

  data = cu.torch_load(path, map_location=torch.device('cpu'))

  os.makedirs(dest_folder)

  ctx = pyu.make_object(seqidx=0)
  xdata = expand(data, dest_folder, ctx)

  pyu.write_config(xdata, os.path.join(dest_folder, 'index.yaml'))


def reload(xdata, stg_folder):
  if isinstance(xdata, collections.abc.Mapping):
    data = dict()
    for name, value in xdata.items():
      data[name] = reload(value, stg_folder)

    return data
  else:
    lpath = os.path.join(stg_folder, xdata)

    return cu.torch_load(lpath, map_location=torch.device('cpu'))


def load_data(path):
  alog.debug(f'Loading {path} checkpoint')

  xdata = pyu.load_config(path)

  return reload(xdata, os.path.dirname(path))


def do_extract(args):
  extract_data(args.input, args.output)


def do_reload(args):
  data = load_data(args.input)

  torch.save(data, args.output)


if __name__ == '__main__':
  parser = argparse.ArgumentParser('Manipulate parameter names from PyTorch checkpoints')
  parser.add_argument('--log_level', default='DEBUG')

  subparsers = parser.add_subparsers(required=True,
                                     help='Command help')

  extract_parser = subparsers.add_parser(
    'extract',
    help='Extracts checkpoint data to a destination folder')
  extract_parser.add_argument('--input', required=True)
  extract_parser.add_argument('--output', required=True)
  extract_parser.set_defaults(cmd_fn=do_extract)

  reload_parser = subparsers.add_parser(
    'reload',
    help='Reloads data from an extract folder and reassemble the data')
  reload_parser.add_argument('--input', required=True)
  reload_parser.add_argument('--output', required=True)
  reload_parser.set_defaults(cmd_fn=do_reload)

  args = parser.parse_args()

  alog.basic_setup(log_level=args.log_level)

  args.cmd_fn(args)

