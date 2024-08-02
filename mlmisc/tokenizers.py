import array
import io
import os

import sentencepiece as spm
import torch

from py_misc_utils import alog


def create_tokenizer(path, max_vocab_size,
                     proto_path=None,
                     model_type=None,
                     **kwargs):
  if proto_path is not None and os.path.isfile(proto_path):
    with open(proto_path, mode='rb') as f:
      proto_data = f.read()
  else:
    spstg = io.BytesIO()
    spm.SentencePieceTrainer.train(input=path,
                                   model_writer=spstg,
                                   model_type=model_type or 'bpe',
                                   vocab_size=max_vocab_size,
                                   **kwargs)

    proto_data = spstg.getvalue()

    if proto_path is not None:
      with open(proto_path, mode='wb') as f:
        f.write(proto_data)

  toknz = spm.SentencePieceProcessor(model_proto=proto_data)

  return toknz


def enum_chunks(path, chunk_size=None):
  chunk_size = chunk_size or 50 * 1024 * 1024
  with open(path, mode='rb') as f:
    pos, rem = 0, b''
    while True:
      alog.debug0(f'Reading from {f.tell()}')

      rdata = f.read(chunk_size)
      data = rem + rdata
      epos = data.rfind(b'\n')
      if epos >= 0:
        rem = data[epos + 1: ]
        data = data[: epos + 1]

      yield data

      if chunk_size > len(rdata):
        break


def tokenize_data(path, toknz, chunk_size=None, dtype=None):
  tokens = array.array('I')
  for chunk in enum_chunks(path, chunk_size=chunk_size):
    enc = toknz.encode(chunk)
    tokens.extend(enc)

  return torch.tensor(tokens, dtype=dtype or torch.long)

