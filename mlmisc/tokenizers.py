import array
import io
import os

import py_misc_utils.alog as alog
import py_misc_utils.file_overwrite as pyfow
import py_misc_utils.fs_utils as pyfsu
import py_misc_utils.gfs as gfs
import py_misc_utils.inspect_utils as pyiu
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


# Make an HuggingFace tokenizer look like a SentencePiece one.
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


def _wrap_tokenizer(tokenizer):
  if isinstance(tokenizer, spm.SentencePieceProcessor):
    return tokenizer
  if isinstance(tokenizer, transformers.tokenization_utils_base.PreTrainedTokenizerBase):
    return FpTokenizerWrapper(tokenizer)

  alog.xraise(ValueError, f'Unknown tokenizer class: {pyiu.qual_name(tokenizer)}')


def from_pretrained(module_path, model_name, **kwargs):
  tclass, = pymu.import_module_names(module_path)
  tokenizer = tclass.from_pretrained(
    model_name,
    use_fast=True,
    trust_remote_code=False,
    cache_dir=gfs.cache_dir(),
    **kwargs)

  return _wrap_tokenizer(tokenizer)


def from_model(path):
  local_path = gfs.as_local(path)

  return spm.SentencePieceProcessor(model_file=local_path)


def from_config(tokenizer_config, **kwargs):
  alog.info(f'Creating tokenizer from "{tokenizer_config}" ...')
  tokenizer = conf.create_object('Tokenizer', tokenizer_config, **kwargs)

  return _wrap_tokenizer(tokenizer)


def from_iterator(data_iter, max_vocab_size,
                  proto_path=None,
                  model_type='bpe',
                  **kwargs):
  spstg = io.BytesIO()
  spm.SentencePieceTrainer.train(sentence_iterator=data_iter,
                                 model_writer=spstg,
                                 model_type=model_type,
                                 vocab_size=max_vocab_size,
                                 **kwargs)

  proto_data = spstg.getvalue()

  if proto_path is not None:
    with pyfow.FileOverwrite(proto_path, mode='wb') as pfd:
      pfd.write(proto_data)

  tokenizer = spm.SentencePieceProcessor(model_proto=proto_data)

  return tokenizer


def create_tokenizer(path, max_vocab_size, proto_path=None, **kwargs):
  if (proto_path is not None and os.path.isfile(proto_path) and
      pyfsu.is_newer_file(proto_path, path)):
    tokenizer = load_tokenizer(proto_path)
    if tokenizer.vocab_size() == max_vocab_size:
      return tokenizer

    alog.warning(f'Existing tokenizer has vocabulary size {tokenizer.vocab_size()} ' \
                 f'but {max_vocab_size} is required. Rebuilding ...')

  chunk_size = 4 * 1024**2

  return from_iterator(enum_chunks(path, chunk_size=chunk_size), max_vocab_size,
                       proto_path=proto_path,
                       max_sentence_length=chunk_size + 4096,
                       **kwargs)


def _chunk_end(data, punct):
  pos = -1
  for c in punct:
    if (pos := data.rfind(c)) >= 0:
      break

  if pos < 0:
    return len(data)

  pos += 1
  while len(data) > pos and punct.find(data[pos]) >= 0:
    pos += 1

  return pos


def enum_chunks(path, chunk_size=4 * 1024**2, binary=True, punct=b'.;?!:,\n'):
  with gfs.open(path, mode='rb') as fd:
    rem = b''
    while True:
      alog.debug0(f'Reading from {path} at offset {fd.tell()}')

      rdata = fd.read(chunk_size)
      data = rem + rdata
      epos = _chunk_end(data, punct)

      rem = data[epos:]
      data = data[: epos]

      yield data if binary else data.decode()

      if chunk_size > len(rdata):
        break


def tokenize_data(path, tokenizer, chunk_size=4 * 1024**2, dtype=None):
  tokens = array.array('I')
  for chunk in enum_chunks(path, chunk_size=chunk_size):
    enc = tokenizer.encode(chunk)
    tokens.extend(enc)

  return torch.tensor(tokens, dtype=dtype or torch.long)

