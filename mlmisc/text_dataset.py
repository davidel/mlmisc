import os

import py_misc_utils.alog as alog
import py_misc_utils.assert_checks as tas
import py_misc_utils.gen_fs as gfs
import py_misc_utils.http_cache as pyhc
import py_misc_utils.uncompress as pyunc
import py_misc_utils.utils as pyu
import torch

from . import dataset_base as dsb
from . import next_sequence_dataset as nsd
from . import next_token_dataset as ntd
from . import tokenizers as tkz
from . import utils as ut


def build_dataset(tokenizer, tokens, split_pct, context_size, is_sequence):
  train_limit = int(len(tokens) * split_pct)
  train_data = tokens[: train_limit]
  test_data = tokens[train_limit:]

  # We used torch.int in tkz.tokenize_data() above to reduce the memory footprint,
  # but some PyTorch APIs require torch.long (!?!) so we convert them on the fly.
  pipeline = dsb.Pipeline()
  pipeline.add(dsb.transformer(target=dsb.to_transform(dtype=torch.long)))

  ds_args = dict(
    pipeline=pipeline,
    tokenizer=tokenizer,
  )
  if is_sequence:
    train_dataset = nsd.NextSequenceDataset(train_data, context_size, **ds_args)
    test_dataset = nsd.NextSequenceDataset(test_data, context_size, **ds_args)
  else:
    train_dataset = ntd.NextTokenDataset(train_data, context_size, **ds_args)
    test_dataset = ntd.NextTokenDataset(test_data, context_size, **ds_args)

  return dict(train=train_dataset, test=test_dataset)


def load(proto_path, tokens_path, context_size,
         is_sequence=None,
         split_pct=None,
         **kwargs):
  is_sequence = pyu.value_or(is_sequence, True)
  split_pct = pyu.value_or(split_pct, 0.9)

  tokenizer = tkz.load_tokenizer(proto_path)
  tokens = ut.torch_load(tokens_path)

  return build_dataset(tokenizer, tokens, split_pct, context_size, is_sequence)


def create(content_path, context_size,
           max_vocab_size=None,
           module_path=None,
           model_name=None,
           cache_dir=None,
           is_sequence=None,
           split_pct=None,
           **kwargs):
  cache_dir = gfs.cache_dir(path=cache_dir)
  is_sequence = pyu.value_or(is_sequence, True)
  split_pct = pyu.value_or(split_pct, 0.9)

  datasets_dir = os.path.join(cache_dir, 'datasets')

  local_content_path = gfs.as_local(content_path, cache_storage=cache_dir)
  with pyunc.Uncompress(local_content_path) as datafile:
    ds_name = os.path.splitext(os.path.basename(datafile))[0]

    if module_path is None:
      ds_dir = os.path.join(datasets_dir, ds_name, 'spm')
      os.makedirs(ds_dir, exist_ok=True)

      tokens_path = os.path.join(ds_dir, 'tokens.pt')
      proto_path = os.path.join(ds_dir, 'tokenizer.proto')
      tokenizer_kwargs = pyu.dict_subset(kwargs, 'pad_id,unk_id,bos_id,eos_id')

      tokenizer = tkz.create_tokenizer(datafile, max_vocab_size,
                                       proto_path=proto_path,
                                       remove_extra_whitespaces=False,
                                       user_defined_symbols=['\n', '\r'],
                                       **tokenizer_kwargs)

      if os.path.isfile(tokens_path) and pyu.is_newer_file(tokens_path, proto_path):
        tokens = ut.torch_load(tokens_path)
      else:
        tokens = tkz.tokenize_data(datafile, tokenizer, dtype=torch.int)
        torch.save(tokens, tokens_path)

      alog.info(f'Tokenizer proto file generated at "{proto_path}"')
    else:
      ds_dir = os.path.join(datasets_dir, ds_name, 'pre_trained')
      os.makedirs(ds_dir, exist_ok=True)

      tokenizer = tkz.from_pretrained(module_path, model_name, cache_dir=cache_dir)

      tokenizer_str = str(tokenizer)
      tokenizer_path = os.path.join(ds_dir, 'tokenizer.repr')
      if os.path.isfile(tokenizer_path):
        with open(tokenizer_path, mode='r') as tfd:
          stored_tokenizer_str = tfd.read()
        needs_tokenization = tokenizer_str == stored_tokenizer_str
      else:
        needs_tokenization = True

      tokens_path = os.path.join(ds_dir, 'tokens.pt')
      if os.path.isfile(tokens_path) and not needs_tokenization:
        tokens = ut.torch_load(tokens_path)
      else:
        tokens = tkz.tokenize_data(datafile, tokenizer, dtype=torch.int)
        torch.save(tokens, tokens_path)
        with open(tokenizer_path, mode='w') as tfd:
          tfd.write(tokenizer_str)

  return build_dataset(tokenizer, tokens, split_pct, context_size, is_sequence)

