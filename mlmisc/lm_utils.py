import torch
import torch.nn as nn
import torch.nn.functional as F

import py_misc_utils.alog as alog


def top_k_logits(logits, k):
  v, _ = torch.topk(logits, k)
  logits[logits < v[:, -1:]] = float('-inf')

  return logits


def create_eval_sequence(iseq, context_size, pad_mode, pad_value):
  ssize = min(context_size, iseq.shape[1])
  if iseq.shape[1] >= context_size:
    seq = iseq[:, -context_size:]
  elif pad_mode == 'none':
    seq = iseq
  else:
    seq = torch.full((1, context_size), pad_value,
                     dtype=iseq.dtype,
                     device=iseq.device)
    if pad_mode == 'front':
      seq[:, -iseq.shape[1]: ] = iseq
    elif pad_mode == 'back':
      seq[:, : iseq.shape[1]] = iseq
    else:
      alog.xraise(ValueError, f'Unknown pad mode: {pad_mode}')

  return seq, ssize


def select_prediction_logits(logits, seqlen):
  # If a model is trained with next next sequence as target (example:
  # 'a b c d' -> 'b c d e') the logits shape is (N, C, V) with N being
  # the batch size, C the context length, and V the vocaboulary size,
  # and the target token is the entry @ (seqlen - 1).
  return logits if logits.dim() <= 2 else logits[:, seqlen - 1, :]


@torch.no_grad()
def generate(model, seq, context_size, steps, pad_mode, pad_value,
             temperature=None,
             sample=False,
             top_k=None):
  iseq = seq.unsqueeze(0) if seq.dim() == 1 else seq

  model.eval()
  for i in range(steps):
    cseq, ssize = create_eval_sequence(iseq, context_size, pad_mode, pad_value)

    y = model(cseq)

    # Handle both model which return the simple output (loss handled externally)
    # and the ones which return the (output, less) tuple.
    logits = y[0] if isinstance(y, (list, tuple)) else y

    logits = select_prediction_logits(logits, ssize)
    if temperature is not None:
      logits /= temperature
    if top_k is not None and top_k > 0:
      logits = top_k_logits(logits, top_k)
    probs = F.softmax(logits, dim=-1)

    if sample:
      next_token = torch.multinomial(probs, num_samples=1)
    else:
      next_token = torch.argmax(probs, dim=-1, keepdim=True)

    iseq = torch.cat((iseq, next_token), dim=1)

  return iseq.squeeze(0) if iseq.dim() > seq.dim() else iseq

