import py_misc_utils.assert_checks as tas
import torch
import torch.nn as nn
from torch.nn import functional as F


def create_diag_mask(size, device='cpu'):
  return torch.tril(torch.ones(size, size, device=device)) == 0


class SelfAttention(nn.Module):

  def __init__(self, embed_size, num_heads, dropout=0, bias=True):
    super().__init__()
    tas.check_eq(embed_size % num_heads, 0,
                 msg=f'Embed size (embed_size) must be multiple of the number of heads ({num_heads})')
    # We stack key, query, value projections in a single matmul.
    self.attn_stack = nn.Linear(embed_size, 3 * embed_size, bias=bias)
    self.output_proj = nn.Linear(embed_size, embed_size, bias=bias)
    self.attn_dropout = nn.Dropout(dropout)
    self.resid_dropout = nn.Dropout(dropout)
    self.num_heads = num_heads
    self.dropout = dropout
    self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

  def forward(self, x, mask=None):
    batch_size, seqlen, embed_size = x.shape
    head_size = embed_size // self.num_heads

    q, k, v  = self.attn_stack(x).split(self.embed_size, dim=2)

    # Output below is (batch_size, num_heads, seqlen, head_size)
    k = k.view(batch_size, seqlen, self.num_heads, head_size).transpose(1, 2)
    q = q.view(batch_size, seqlen, self.num_heads, head_size).transpose(1, 2)
    v = v.view(batch_size, seqlen, self.num_heads, head_size).transpose(1, 2)

    # (batch_size, num_heads, seqlen, head_size) x (batch_size, num_heads, head_size, seqlen)
    #   -> (batch_size, num_heads, seqlen, seqlen)
    if self.flash:
      y = torch.nn.functional.scaled_dot_product_attention(q, k, v,
                                                           attn_mask=None,
                                                           dropout_p=self.dropout if self.training else 0,
                                                           is_causal=True)
    else:
      att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
      if mask is not None:
        att = att.masked_fill(mask, float('-inf'))
      att = F.softmax(att, dim=-1)
      att = self.attn_dropout(att)
      # (batch_size, num_heads, seqlen, seqlen) x (batch_size, num_heads, seqlen, head_size)
      #   -> (batch_size, num_heads, seqlen, head_size)
      y = att @ v
      y = y.transpose(1, 2).contiguous().view(batch_size, seqlen, embed_size)

    y =  self.resid_dropout(self.output_proj(y))

    return y

