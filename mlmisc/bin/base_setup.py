import argparse
import multiprocessing
import os

import mlmisc.core_utils as mlcu
import mlmisc.debug_utils as mldu
import py_misc_utils.gfs as gfs
import torch


def add_parser_arguments(parser):
  parser.add_argument('--cache_dir',
                      default=gfs.cache_dir(),
                      help='The folder used to store cached files')
  parser.add_argument('--device',
                      help='The device to be used')
  parser.add_argument('--seed', type=int,
                      help='The seed for the random number generators')
  parser.add_argument('--cpu_num_threads', type=int,
                      help='The number of threads to dedicate to the PyTorch CPU device')
  parser.add_argument('--nan_debug', action=argparse.BooleanOptionalAction, default=False,
                      help='Enable Autograd anomaly detection')
  parser.add_argument('--mp_start_method',
                      default=os.getenv('MP_START_METHOD', 'default'),
                      choices=('fork', 'forkserver', 'spawn', 'default'),
                      help='Sets the Python multiprocessing start method')


def setup(args):
  if args.mp_start_method != 'default':
    multiprocessing.set_start_method(args.mp_start_method, force=True)
  if args.seed is not None:
    mlcu.randseed(args.seed)
  if args.nan_debug:
    mldu.set_detect_anomaly(True)
  if args.cpu_num_threads is not None:
    torch.set_num_threads(args.cpu_num_threads)
  if args.device is None:
    args.device = mlcu.get_device()
  else:
    args.device = torch.device(args.device)

