import torch
import torch.nn as nn
import torch.nn.functional as F

import py_misc_utils.alog as alog

from . import layer_utils as lu


def build_vocab_head(embed_size, vocab_size,
                     activation='gelu',
                     mid_size_factor=2):
  mid_size = min(mid_size_factor * embed_size, vocab_size)

  return nn.Sequential(
    lu.create(activation),
    nn.Linear(embed_size, mid_size, bias=False),
    nn.LayerNorm(mid_size),
    lu.create(activation),
    nn.Linear(mid_size, vocab_size, bias=False),
  )


def mask_top_k_logits(logits, k):
  v, _ = torch.topk(logits, k)
  logits[logits < v[..., -1:]] = float('-inf')

  return logits


def create_eval_sequence(iseq, context_size, pad_mode, pad_value):
  seqlen = iseq.shape[1]
  ssize = min(context_size, seqlen)
  if seqlen >= context_size:
    seq = iseq[..., -context_size:]
  elif pad_mode == 'none':
    seq = iseq
  else:
    seq = torch.full((iseq.shape[0], context_size), pad_value,
                     dtype=iseq.dtype,
                     device=iseq.device)
    if pad_mode == 'front':
      seq[..., -seqlen:] = iseq
    elif pad_mode == 'back':
      seq[..., : seqlen] = iseq
    else:
      alog.xraise(ValueError, f'Unknown pad mode: {pad_mode}')

  return seq, ssize


def select_prediction_logits(logits, seqlen):
  # If a model is trained with next next sequence as target (example:
  # 'a b c d' -> 'b c d e') the logits shape is (N, C, V) with N being
  # the batch size, C the context length, and V the vocaboulary size,
  # and the target token is the entry @ (seqlen - 1).
  return logits if logits.ndim <= 2 else logits[..., seqlen - 1, :]


def generate(evalfn, seq, context_size, steps, pad_mode, pad_value,
             temperature=None,
             sample=True,
             top_k=None):
  # Add artificial batch dimension.
  iseq = seq.unsqueeze(0)

  for i in range(steps):
    cseq, ssize = create_eval_sequence(iseq, context_size, pad_mode, pad_value)

    logits = evalfn(cseq)

    logits = select_prediction_logits(logits, ssize)
    if temperature is not None:
      logits /= temperature
    if top_k is not None and top_k > 0:
      logits = mask_top_k_logits(logits, top_k)
    probs = F.softmax(logits, dim=-1)

    if sample:
      next_token = torch.multinomial(probs, num_samples=1)
    else:
      next_token = torch.argmax(probs, dim=-1, keepdim=True)

    iseq = torch.cat((iseq, next_token), dim=1)

  return iseq.squeeze(0)

