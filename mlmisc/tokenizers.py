import array
import io
import os

import py_misc_utils.alog as alog
import py_misc_utils.utils as pyu
import sentencepiece as spm
import torch


def load_tokenizer(proto_path):
  with open(proto_path, mode='rb') as pfd:
    proto_data = pfd.read()

  toknz = spm.SentencePieceProcessor(model_proto=proto_data)

  return toknz


def create_tokenizer(path, max_vocab_size,
                     proto_path=None,
                     model_type=None,
                     **kwargs):
  if (proto_path is not None and os.path.isfile(proto_path) and
      pyu.is_newer_file(proto_path, path)):
    toknz = load_tokenizer(proto_path)
    if toknz.vocab_size() == max_vocab_size:
      return toknz

    alog.warning(f'Existing tokenizer has vocabulary size {toknz.vocab_size()} ' \
                 f'but {max_vocab_size} is required. Rebuilding ...')

  spstg = io.BytesIO()
  spm.SentencePieceTrainer.train(input=path,
                                 model_writer=spstg,
                                 model_type=model_type or 'bpe',
                                 vocab_size=max_vocab_size,
                                 **kwargs)

  proto_data = spstg.getvalue()

  if proto_path is not None:
    with open(proto_path, mode='wb') as pfd:
      pfd.write(proto_data)

  toknz = spm.SentencePieceProcessor(model_proto=proto_data)

  return toknz


def enum_chunks(path, chunk_size=None):
  chunk_size = chunk_size or 32 * 1024 * 1024
  with open(path, mode='rb') as fd:
    rem = b''
    while True:
      alog.debug0(f'Reading from {path} at offset {fd.tell()}')

      rdata = fd.read(chunk_size)
      data = rem + rdata
      if (epos := data.rfind(b'\n')) < 0:
        epos = data.rfind(b' ')
      if epos >= 0:
        rem = data[epos + 1: ]
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


def tokenize_data(path, toknz, chunk_size=None, dtype=None):
  tokens = array.array('I')
  for chunk in enum_chunks(path, chunk_size=chunk_size):
    enc = toknz.encode(chunk)
    tokens.extend(enc)

  return torch.tensor(tokens, dtype=dtype or torch.long)

