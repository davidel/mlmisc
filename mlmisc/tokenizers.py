import array
import io
import os

import py_misc_utils.alog as alog
import py_misc_utils.file_overwrite as pyfow
import py_misc_utils.fs_utils as pyfsu
import py_misc_utils.gfs as gfs
import py_misc_utils.module_utils as pymu
import py_misc_utils.utils as pyu
import sentencepiece as spm
import torch

from . import config as conf


def load_tokenizer(proto_path):
  alog.debug(f'Loading tokenizer from "{proto_path}"')

  with gfs.open(proto_path, mode='rb') as pfd:
    proto_data = pfd.read()

  tokenizer = spm.SentencePieceProcessor(model_proto=proto_data)

  return tokenizer


class FpTokenizerWrapper:

  def __init__(self, tokenizer):
    self._tokenizer = tokenizer

  def vocab_size(self):
    return self._tokenizer.vocab_size

  def bos_id(self):
    return self._tokenizer.bos_token_id

  def eos_id(self):
    return self._tokenizer.eos_token_id

  def unk_id(self):
    return self._tokenizer.unk_token_id

  def encode(self, data):
    edata = data if isinstance(data, str) else data.decode()

    return self._tokenizer.encode(edata,
                                  add_special_tokens=False,
                                  verbose=False)

  def decode(self, data):
    return self._tokenizer.decode(data)


def from_pretrained(module_path, model_name, cache_dir=None, **kwargs):
  cache_dir = gfs.cache_dir(path=cache_dir)

  tclass, = pymu.import_module_names(module_path)
  tokenizer = tclass.from_pretrained(
    model_name,
    use_fast=True,
    trust_remote_code=False,
    cache_dir=cache_dir,
    **kwargs)

  return FpTokenizerWrapper(tokenizer)


def from_config(tokenizer_config, **kwargs):
  tokenizer = conf.create_object('Tokenizer', tokenizer_config, **kwargs)

  return FpTokenizerWrapper(tokenizer)


def create_tokenizer(path, max_vocab_size,
                     proto_path=None,
                     model_type=None,
                     **kwargs):
  if (proto_path is not None and os.path.isfile(proto_path) and
      pyfsu.is_newer_file(proto_path, path)):
    tokenizer = load_tokenizer(proto_path)
    if tokenizer.vocab_size() == max_vocab_size:
      return tokenizer

    alog.warning(f'Existing tokenizer has vocabulary size {tokenizer.vocab_size()} ' \
                 f'but {max_vocab_size} is required. Rebuilding ...')

  spstg = io.BytesIO()
  spm.SentencePieceTrainer.train(input=path,
                                 model_writer=spstg,
                                 model_type=model_type or 'bpe',
                                 vocab_size=max_vocab_size,
                                 **kwargs)

  proto_data = spstg.getvalue()

  if proto_path is not None:
    with pyfow.FileOverwrite(proto_path, mode='wb') as pfd:
      pfd.write(proto_data)

  tokenizer = spm.SentencePieceProcessor(model_proto=proto_data)

  return tokenizer


def enum_chunks(path, chunk_size=32 * 1024 * 1024):
  with gfs.open(path, mode='rb') as fd:
    rem = b''
    while True:
      alog.debug0(f'Reading from {path} at offset {fd.tell()}')

      rdata = fd.read(chunk_size)
      data = rem + rdata
      if (epos := data.rfind(b'\n')) < 0:
        epos = data.rfind(b' ')
      if epos >= 0:
        rem = data[epos + 1:]
        data = data[: epos + 1]
      else:
        # We did not find an EOL (or alternatively a space) within the read buffer.
        # If the buffer is big enough, this should mean we are at the end of the
        # data, otherwise we end up feeding a truncated word in.
        # This should not happen though, given a big enough buffer and data being
        # actually text!
        rem = b''

      yield data

      if chunk_size > len(rdata):
        break


def tokenize_data(path, tokenizer, chunk_size=None, dtype=None):
  tokens = array.array('I')
  for chunk in enum_chunks(path, chunk_size=chunk_size):
    enc = tokenizer.encode(chunk)
    tokens.extend(enc)

  return torch.tensor(tokens, dtype=dtype or torch.long)

