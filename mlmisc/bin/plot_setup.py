import argparse
import os

import matplotlib.pyplot as plt
import mlmisc.utils as mlut
import py_misc_utils.gen_fs as gfs


def add_parser_arguments(parser, skip_output=False):
  parser.add_argument('--dpi', type=int, default=128,
                      help='The DPI to be used to generate the image')
  parser.add_argument('--img_h', type=int, default=6,
                      help='The height of the plot image')
  parser.add_argument('--grid',
                      help='The style of the plot grid')
  parser.add_argument('--aspect', default='4/3',
                      help='The aspect ratio to be used to generate the image')
  if not skip_output:
    parser.add_argument('--plotfile',
                        help='The path to the file where the plot image should be saved')
    parser.add_argument('--show', action=argparse.BooleanOptionalAction,
                        default=False,
                        help='Show the image of screen')

def plot_args(args):
  aspect = eval(args.aspect)
  img_w = int(args.img_h * aspect)

  return dict(figsize=(img_w, args.img_h), dpi=args.dpi)


def plot_setup(args, ax):
  if args.grid:
    ax.grid(linestyle=args.grid)


def plot(args):
  if args.plotfile:
    imgid = getattr(args, '_imgid', 0)
    if imgid > 0:
      bpath, ext = os.path.spliext(args.plotfile)
      path = f'{bpath}-{imgid + 1}{ext}'
    else:
      path = args.plotfile

    with gfs.open(path, mode='wb') as imgfd:
      plt.savefig(imgfd)

    setattr(args, '_imgid', imgid + 1)

  if args.show:
    plt.show()


def setup(args):
  pass

