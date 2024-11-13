import mlmisc.dataset_utils as mldu
import py_misc_utils.alog as alog
import py_misc_utils.gen_fs as gfs
import py_misc_utils.utils as pyu
import torch
import torch.nn as nn


def add_parser_arguments(parser):
  parser.add_argument('--dataset', required=True,
                      help='The name of the dataset to be used')
  parser.add_argument('--dataset_transform',
                      help='The path to the Python file containing the TRAIN_TRANS ' \
                      'and TEST_TRANS dataset transformations')
  parser.add_argument('--dataset_key_selector',
                      help='The comma-separated keys to be used to select the dataset items')
  parser.add_argument('--dataset_index_selector',
                      help='The comma-separated integer indices to be used to select the dataset items')
  parser.add_argument('--dataset_kwargs',
                      help='The comma-separated key=value arguments for the dataset constructor ' \
                      f'(or the path to a YAML/JSON file containing such configuration)')
  parser.add_argument('--show_images', type=int,
                      help='The number of sample images to be shown before starting training')


def create_dataset(args):
  train_trans = test_trans = tgt_train_trans = tgt_test_trans = nn.Identity()
  if args.dataset_transform:
    with gfs.open(args.dataset_transform, mode='r') as dtf:
      code = dtf.read()

    syms = 'TRAIN_TRANS,TEST_TRANS,TGT_TRAIN_TRANS,TGT_TEST_TRANS'
    train_trans, test_trans, tgt_train_trans, tgt_test_trans = pyu.compile(code, syms)

    tgt_train_trans = tgt_train_trans or nn.Identity()
    tgt_test_trans = tgt_test_trans or nn.Identity()

    alog.info(f'Train Dataset Transforms:\n{train_trans}')
    alog.info(f'Train Dataset Target Transforms:\n{tgt_train_trans}')
    alog.info(f'Test Dataset Transforms:\n{test_trans}')
    alog.info(f'Test Dataset Target Transforms:\n{tgt_test_trans}')

  if args.dataset_key_selector:
    select_fn = mldu.items_selector(pyu.comma_split(args.dataset_key_selector))
  elif args.dataset_index_selector:
    indices = [int(i) for i in pyu.comma_split(args.dataset_index_selector)]
    select_fn = mldu.items_selector(indices)
  else:
    select_fn = None

  dataset_kwargs = pyu.parse_config(args.dataset_kwargs) if args.dataset_kwargs else dict()
  cache_dir = gfs.cache_dir(path=getattr(args, 'cache_dir', None))

  dsets = mldu.create_dataset(args.dataset,
                              cache_dir=cache_dir,
                              select_fn=select_fn,
                              transform=dict(train=train_trans, test=test_trans),
                              target_transform=dict(train=tgt_train_trans, test=tgt_test_trans),
                              dataset_kwargs=dataset_kwargs)

  train_dataset = dsets['train']
  test_dataset = dsets['test']

  alog.info(f'Train/Test Dataset samples = {len(train_dataset)}/{len(test_dataset)}')

  if args.show_images:
    mldu.show_images(train_dataset, args.show_images)

  return train_dataset, test_dataset

