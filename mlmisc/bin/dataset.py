import mlmisc.dataset_base as mldb
import mlmisc.dataset_utils as mldu
import py_misc_utils.alog as alog
import py_misc_utils.dynamod as pydm
import py_misc_utils.fs_utils as pyfsu
import py_misc_utils.gfs as gfs
import py_misc_utils.utils as pyu
import torch
import torch.nn as nn


def add_parser_arguments(parser):
  parser.add_argument('--dataset', required=True,
                      help='The name of the dataset to be used')
  parser.add_argument('--dataset_transform',
                      help='The path to the Python file containing the TRAIN_TRANS ' \
                      'and TEST_TRANS dataset transformations')
  parser.add_argument('--dataset_selector',
                      help='The path to the Python file containing the SELECTOR function implementation')
  parser.add_argument('--dataset_key_selector',
                      help='The comma-separated keys to be used to select the dataset items')
  parser.add_argument('--dataset_index_selector',
                      help='The comma-separated integer indices to be used to select the dataset items')
  parser.add_argument('--dataset_kwargs',
                      help='The comma-separated key=value arguments for the dataset constructor ' \
                      f'(or the path to a YAML/JSON file containing such configuration)')
  parser.add_argument('--show_images', type=int,
                      help='The number of sample images to be shown before starting training')


def _comma_split(value):
  # If the split contains a single value it is returned as scalar.
  # To return a single value as list, use "VALUE,", like in Python one would tuples.
  tokens = pyu.comma_split(value)
  if len(tokens) == 1:
    return tokens[0]
  elif tokens[-1] == '':
    return tokens[: -1]
  else:
    return tokens


def create_dataset(args):
  if args.dataset_selector:
    code = pyfsu.readall(args.dataset_selector).decode()
    module = pydm.create_module('mlmisc.dataset.dataset_selector', code)

    select_fn = getattr(module, 'SELECTOR', None)
  elif args.dataset_key_selector:
    select_fn = mldb.items_selector(_comma_split(args.dataset_key_selector))
  elif args.dataset_index_selector:
    sres = _comma_split(args.dataset_index_selector)
    index = [int(i) for i in sres] if isinstance(sres, (list, tuple)) else int(sres)
    select_fn = mldb.items_selector(index)
  else:
    select_fn = None

  train_trans = test_trans = tgt_train_trans = tgt_test_trans = nn.Identity()
  if args.dataset_transform:
    code = pyfsu.readall(args.dataset_transform).decode()
    module = pydm.create_module('mlmisc.dataset.dataset_transform', code)

    syms = 'TRAIN_TRANS,TEST_TRANS,TGT_TRAIN_TRANS,TGT_TEST_TRANS'
    train_trans, test_trans, tgt_train_trans, tgt_test_trans = (
      getattr(module, name, None) for name in pyu.comma_split(syms))

    tgt_train_trans = tgt_train_trans or nn.Identity()
    tgt_test_trans = tgt_test_trans or nn.Identity()

    alog.info(f'Train Dataset Transforms:\n{train_trans}')
    alog.info(f'Train Dataset Target Transforms:\n{tgt_train_trans}')
    alog.info(f'Test Dataset Transforms:\n{test_trans}')
    alog.info(f'Test Dataset Target Transforms:\n{tgt_test_trans}')

  dataset_kwargs = pyu.parse_config(args.dataset_kwargs) if args.dataset_kwargs else dict()
  alog.debug0(f'Dataset Args: {dataset_kwargs}')

  dsets = mldu.create_dataset(args.dataset,
                              cache_dir=gfs.cache_dir(),
                              select_fn=select_fn,
                              transform=dict(train=train_trans, test=test_trans),
                              target_transform=dict(train=tgt_train_trans, test=tgt_test_trans),
                              dataset_kwargs=dataset_kwargs)

  train_dataset = dsets['train']
  test_dataset = dsets['test']

  train_size, test_size = mldu.dataset_size(train_dataset), mldu.dataset_size(test_dataset)
  if train_size is not None and test_size is not None:
    alog.info(f'Train/Test Dataset samples = {train_size}/{test_size}')

  if args.show_images:
    mldu.show_images(train_dataset, args.show_images)

  return train_dataset, test_dataset

