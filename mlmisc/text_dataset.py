import os

import py_misc_utils.alog as alog
import py_misc_utils.assert_checks as tas
import py_misc_utils.fs_utils as pyfsu
import py_misc_utils.gfs as gfs
import py_misc_utils.pipeline as pypl
import py_misc_utils.uncompress as pyunc
import py_misc_utils.utils as pyu
import torch

from . import config as conf
from . import core_utils as cu
from . import dataset_adapters as dsad
from . import dataset_base as dsb
from . import sequence_datasets as seqds
from . import tokenizers as tkz
from . import web_dataset as wds


def build_dataset(tokenizer, tokens, train_pct, context_size, mode):
  train_limit = int(len(tokens) * train_pct)
  train_data = tokens[: train_limit]
  test_data = tokens[train_limit:]

  # We used torch.int in tkz.tokenize_data() to reduce the memory footprint, but
  # some PyTorch APIs require torch.long (!?!) so we convert them on the fly.
  to_long = dsb.to_transform(dtype=torch.long)
  pipeline = pypl.Pipeline(dsb.transformer(sample=to_long, target=to_long))

  ds_args = dict(
    pipeline=pipeline,
    tokenizer=tokenizer,
  )

  train_dataset = seqds.SequenceDataset(train_data, context_size, mode,
                                        **ds_args)
  test_dataset = seqds.SequenceDataset(test_data, context_size, mode,
                                       **ds_args)

  return dict(train=train_dataset, test=test_dataset)


def load(proto_path, tokens_path, context_size, mode,
         train_pct=0.9,
         **kwargs):
  tokenizer = tkz.load_tokenizer(proto_path)
  tokens = cu.torch_load(tokens_path)

  return build_dataset(tokenizer, tokens, train_pct, context_size, mode)


def create(content_path, context_size, mode,
           max_vocab_size=None,
           module_path=None,
           model_name=None,
           train_pct=0.9,
           **kwargs):
  cache_dir = gfs.cache_dir()
  datasets_dir = os.path.join(cache_dir, 'datasets')

  local_content_path = gfs.as_local(content_path)
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

      if os.path.isfile(tokens_path) and pyfsu.is_newer_file(tokens_path, proto_path):
        tokens = cu.torch_load(tokens_path)
      else:
        tokens = tkz.tokenize_data(datafile, tokenizer, dtype=torch.int)
        torch.save(tokens, tokens_path)

      alog.info(f'Tokenizer proto file generated at "{proto_path}"')
    else:
      ds_dir = os.path.join(datasets_dir, ds_name, 'pre_trained')
      os.makedirs(ds_dir, exist_ok=True)

      tokenizer = tkz.from_pretrained(module_path, model_name)

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
        tokens = cu.torch_load(tokens_path)
      else:
        tokens = tkz.tokenize_data(datafile, tokenizer, dtype=torch.int)
        torch.save(tokens, tokens_path)
        with open(tokenizer_path, mode='w') as tfd:
          tfd.write(tokenizer_str)

  return build_dataset(tokenizer, tokens, train_pct, context_size, mode)


def web_create(url, tokenizer_config, field_selector, context_size, mode,
               **kwargs):
  alog.debug(f'Creating web dataset from "{url}" ...')
  dataset = wds.create(url, **kwargs)

  tokenizer = tkz.from_config(tokenizer_config)

  to_long = dsb.to_transform(dtype=torch.long)

  webds = dict()
  for kind, dset in dataset.items():
    pipeline = pypl.Pipeline(
      dsb.items_selector(field_selector),
      seqds.SequenceProcessor(context_size, mode, tokenizer),
      dsb.transformer(sample=to_long, target=to_long),
    )

    webds[kind] = dsad.IterableTransformDataset(dset, pipeline,
                                                tokenizer=tokenizer)

  return webds

