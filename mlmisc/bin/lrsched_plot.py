import argparse

import matplotlib.pyplot as plt
import mlmisc.config as mlco
import mlmisc.core_utils as mlcu
import py_misc_utils.alog as alog
import py_misc_utils.app_main as pyam
import py_misc_utils.gfs as gfs
import py_misc_utils.utils as pyu
import torch
import torch.nn as nn
import torch.optim as optim

from . import plot_setup as ps


def format_scheduler(lr_scheduler):
  cpath, params = lr_scheduler.split(':', maxsplit=1)

  return '\n'.join((cpath, params))


def main(args):
  ps.setup(args)

  # Dummy net and trivial SGD optimizer to feed to the LR scheduler.
  net = nn.Linear(8, 8)
  optimizer = optim.SGD(net.parameters(), lr=args.init_lr)
  optimizer.zero_grad()

  scheduler = mlco.create_lr_scheduler(optimizer, args.lr_scheduler)

  lrs = []
  for n in range(args.num_steps):
    optimizer.step()
    scheduler.step()
    lrs.append(mlcu.get_lr(optimizer))

  fig, ax = plt.subplots(**ps.plot_args(args))

  ps.plot_setup(args, ax)
  ax.plot(tuple(range(args.num_steps)), lrs)

  ax.set(xlabel='Step Number',
         ylabel='Learning Rate',
         title=format_scheduler(args.lr_scheduler))
  ps.plot(args)


if __name__ == '__main__':
  # Do basic logging setup ... will be setup later once the modules are parsed.
  alog.basic_setup()

  parser = argparse.ArgumentParser(description='LR Scheduler Plotter',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  ps.add_parser_arguments(parser)

  parser.add_argument('--lr_scheduler', required=True,
                      help='The configuration for the learning rate scheduler ' \
                      '(class_path:arg0,...,name0=value0,...)')
  parser.add_argument('--init_lr', type=float, default=1e-2,
                      help='The LR used to initialize the optimizer')
  parser.add_argument('--num_steps', type=int, default=100,
                      help='The number of steps to plot')

  pyam.main(parser, main)

