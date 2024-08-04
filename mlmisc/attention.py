import torch
import torch.nn as nn

import py_misc_utils.assert_checks as tas


class Attention(nn.Module):

  def __init__(self, n_embd, n_head,
               attn_dropout=None,
               dropout=None):
    tas.check_eq(n_embd % n_head, 0,
                 msg=f'Embedding dimension ({n_embd}) must be multiple ' \
                 f'of the number of heads ({n_head})')

    super().__init__()
    self.n_head = n_head
    self.k_prj = nn.Linear(n_embd, n_embd)
    self.q_prj = nn.Linear(n_embd, n_embd)
    self.v_prj = nn.Linear(n_embd, n_embd)
    self.unifyheads = nn.Linear(n_embd, n_embd)

    self.attn_drop = nn.Dropout(attn_dropout or 0.1)
    self.resid_drop = nn.Dropout(dropout or 0.1)

  def forward(self, k, q, v, mask=None):
    b, t, c = k.shape
    h, ch = self.n_head, c // self.n_head

    # Move all to the (B, H, T, CH) shape.
    keys = self.k_prj(k).view(b, t, h, ch).transpose(1, 2)
    queries = self.q_prj(q).view(b, t, h, ch).transpose(1, 2)
    values = self.v_prj(v).view(b, t, h, ch).transpose(1, 2)

    # (B, H, T, CH) @ (B, H, CH, T) => (B, H, T, T)
    att = queries @ keys.transpose(-2, -1)
    if mask is not None:
      att = att.masked_fill(mask[:, :, :t, :t], float('-inf'))
    att = F.softmax(att * ch**-0.5, dim=-1)
    att = self.attn_drop(att)

    # (B, H, T, T) @ (B, H, T, CH) => (B, H, T, CH)
    out = att @ values
    # 1) (B, H, T, CH) => (B, T, H, CH)
    # 2) (B, T, H, CH) => (B, T, H * CH) == (B, T, C)
    out = out.transpose(1, 2).contiguous().view(b, t, c)
    # (B, T, C) @ (C, C) => (B, T, C)
    out = self.unifyheads(out)
    out = self.resid_drop(out)

    return out

