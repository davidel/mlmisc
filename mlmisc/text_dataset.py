import os

import py_misc_utils.alog as alog
import py_misc_utils.assert_checks as tas
import py_misc_utils.utils as pyu
import torch

from . import next_sequence_dataset as nsd
from . import next_token_dataset as ntd
from . import tokenizers as tkz
from . import utils as ut


def create(datafile, context_size, max_vocab_size,
           cache_dir=None,
           is_sequence=None,
           split_pct=None):
  cache_dir = cache_dir or os.path.join(os.getenv('HOME', '.'), 'datasets')
  is_sequence = True if is_sequence is None else is_sequence
  split_pct = 0.9 if split_pct is None else split_pct

  ds_name = os.path.splitext(os.path.basename(datafile))[0]
  ds_dir = os.path.join(cache_dir, ds_name)
  os.makedirs(ds_dir, exist_ok=True)

  proto_path = os.path.join(ds_dir, 'tokenizer.proto')

  tokenizer = tkz.create_tokenizer(datafile, max_vocab_size,
                                   proto_path=proto_path,
                                   remove_extra_whitespaces=False,
                                   user_defined_symbols=['\n', '\r'])

  tokens_path = os.path.join(ds_dir, 'tokens.pt')
  if os.path.isfile(tokens_path) and pyu.is_newer_file(tokens_path, proto_path):
    tokens = ut.torch_load(tokens_path)
  else:
    tokens = tkz.tokenize_data(datafile, tokenizer)
    torch.save(tokens, tokens_path)

  train_limit = int(len(tokens) * split_pct)
  train_data = tokens[: train_limit]
  test_data = tokens[train_limit: ]

  if is_sequence:
    train_dataset = nsd.NextSequenceDataset(train_data, context_size)
    test_dataset = nsd.NextSequenceDataset(test_data, context_size)
  else:
    train_dataset = ntd.NextTokenDataset(train_data, context_size)
    test_dataset = ntd.NextTokenDataset(test_data, context_size)

  return dict(train=train_dataset, test=test_dataset)

