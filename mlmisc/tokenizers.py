import io

import sentencepiece as spm


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

  sp = spm.SentencePieceProcessor(model_proto=proto_data)

  return sp

