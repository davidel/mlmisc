import argparse

import matplotlib.pyplot as plt
import mlmisc.config as mlco
import mlmisc.utils as mlut
import py_misc_utils.alog as alog
import py_misc_utils.app_main as pyam
import py_misc_utils.gen_fs as gfs
import py_misc_utils.utils as pyu
import torch
import torch.nn as nn
import torch.optim as optim


def main(args):
  # Dummy net and trivial SGD optimizer to feed to the LR scheduler.
  net = nn.Linear(8, 8)
  optimizer = optim.SGD(net.parameters(), lr=args.init_lr)
  optimizer.zero_grad()

  scheduler = mlco.create_lr_scheduler(optimizer, args.lr_scheduler)

  lrs = []
  for n in range(args.num_steps):
    optimizer.step()
    scheduler.step()
    lrs.append(mlut.get_lr(optimizer))

  img_w = int(args.img_h * args.aspect)
  fig, ax = plt.subplots(figsize=(img_w, args.img_h), dpi=args.dpi)

  ax.plot(tuple(range(args.num_steps)), lrs)

  ax.set(xlabel='Step Number',
         ylabel='Learning Rate',
         title=f'LR Plot for : {args.lr_scheduler}')
  ax.grid(linestyle=args.grid)

  if args.plotfile:
    with gfs.open(args.plotfile, mode='wb') as imgfd:
      plt.savefig(imgfd)

  if args.show:
    plt.show()


if __name__ == '__main__':
  # Do basic logging setup ... will be setup later once the modules are parsed.
  alog.basic_setup()

  parser = argparse.ArgumentParser(description='LR Scheduler Plotter',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('--lr_scheduler', required=True,
                      help='The configuration for the learning rate scheduler ' \
                      '(class_path:arg0,...,name0=value0,...)')
  parser.add_argument('--init_lr', type=float, default=1e-2,
                      help='The LR used to initialize the optimizer')
  parser.add_argument('--num_steps', type=int, default=100,
                      help='The number of steps to plot')
  parser.add_argument('--dpi', type=int, default=128,
                      help='The DPI to be used to generate the image')
  parser.add_argument('--img_h', type=int, default=6,
                      help='The height of the plot image')
  parser.add_argument('--grid', default='dotted',
                      help='The style of the plot grid')
  parser.add_argument('--aspect', type=float, default=4 / 3,
                      help='The aspect ratio to be used to generate the image')
  parser.add_argument('--plotfile',
                      help='The path to the file where the plot image should be saved')
  parser.add_argument('--show', action=argparse.BooleanOptionalAction,
                      default=False,
                      help='Show the image of screen')

  pyam.main(parser, main)

