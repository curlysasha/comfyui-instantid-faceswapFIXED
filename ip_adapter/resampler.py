# modified from https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/src/helpers.py
import math
import os

import torch
import torch.nn as nn

MAX_FP16_VALUE = 65504.0  # Largest finite fp16 value, safe clamp target
DEBUG_RESAMPLER = os.getenv("INSTANTID_DEBUG_RESAMPLER", "0") == "1"

def _log_tensor_stats(label, tensor):
  if not DEBUG_RESAMPLER:
    return
  print(f"[Resampler Debug] {label}: min={tensor.min().item():.2f}, max={tensor.max().item():.2f}, mean={tensor.mean().item():.2f}, has_nan={torch.isnan(tensor).any().item()}, has_inf={torch.isinf(tensor).any().item()}")

def _sanitize_tensor(label, tensor):
  has_nan = torch.isnan(tensor).any().item()
  has_inf = torch.isinf(tensor).any().item()
  if has_nan or has_inf:
    print(f"[Resampler WARNING] {label}: NaN={has_nan}, Inf={has_inf}. Replacing with finite values.")
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=MAX_FP16_VALUE, neginf=-MAX_FP16_VALUE)
  tensor = torch.clamp(tensor, min=-MAX_FP16_VALUE, max=MAX_FP16_VALUE)
  return tensor

# FFN
class FeedForward(nn.Module):
  def __init__(self, dim, mult=4):
    super().__init__()
    inner_dim = int(dim * mult)
    self.norm = nn.LayerNorm(dim)
    self.linear1 = nn.Linear(dim, inner_dim, bias=False)
    self.gelu = nn.GELU()
    self.linear2 = nn.Linear(inner_dim, dim, bias=False)

  def forward(self, x):
    x = self.norm(x)
    x = self.linear1(x)
    x = torch.clamp(x, min=-100, max=100)  # Prevent overflow after first linear
    x = self.gelu(x)
    x = self.linear2(x)
    x = torch.clamp(x, min=-100, max=100)  # Prevent overflow after second linear
    return x

def reshape_tensor(x, heads):
  bs, length, _ = x.shape
  #(bs, length, width) --> (bs, length, n_heads, dim_per_head)
  x = x.view(bs, length, heads, -1)
  # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
  x = x.transpose(1, 2)
  # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
  x = x.reshape(bs, heads, length, -1)
  return x


class PerceiverAttention(nn.Module):
  def __init__(self, *, dim, dim_head=64, heads=8):
    super().__init__()
    self.scale = dim_head**-0.5
    self.dim_head = dim_head
    self.heads = heads
    inner_dim = dim_head * heads

    self.norm1 = nn.LayerNorm(dim)
    self.norm2 = nn.LayerNorm(dim)

    self.to_q = nn.Linear(dim, inner_dim, bias=False)
    self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
    self.to_out = nn.Linear(inner_dim, dim, bias=False)

  def forward(self, x, latents):
    """
    Args:
        x (torch.Tensor): image features
            shape (b, n1, D)
        latent (torch.Tensor): latent features
            shape (b, n2, D)
    """
    x = self.norm1(x)
    latents = self.norm2(latents)

    b, l, _ = latents.shape

    q = self.to_q(latents)
    kv_input = torch.cat((x, latents), dim=-2)
    k, v = self.to_kv(kv_input).chunk(2, dim=-1)

    q = reshape_tensor(q, self.heads)
    k = reshape_tensor(k, self.heads)
    v = reshape_tensor(v, self.heads)

    # attention
    scale = 1 / math.sqrt(math.sqrt(self.dim_head))
    weight = (q * scale) @ (k * scale).transpose(-2, -1) # More stable with f16 than dividing afterwards
    weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
    out = weight @ v

    out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

    return self.to_out(out)


class Resampler(nn.Module):
  def __init__(
    self,
    dim=1024,
    depth=8,
    dim_head=64,
    heads=16,
    num_queries=8,
    embedding_dim=768,
    output_dim=1024,
    ff_mult=4,
  ):
    super().__init__()

    self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)

    self.proj_in = nn.Linear(embedding_dim, dim)

    self.proj_out = nn.Linear(dim, output_dim)
    self.norm_out = nn.LayerNorm(output_dim)

    self.layers = nn.ModuleList([])
    for _ in range(depth):
      self.layers.append(
        nn.ModuleList(
          [
            PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
            FeedForward(dim=dim, mult=ff_mult),
          ]
        )
      )

  def forward(self, x):
    _log_tensor_stats("Input", x)
    x = _sanitize_tensor("Input", x)

    latents = self.latents.to(x.device).repeat(x.size(0), 1, 1)
    _log_tensor_stats("Latents init", latents)
    latents = _sanitize_tensor("Latents init", latents)

    x = self.proj_in(x)
    _log_tensor_stats("After proj_in", x)
    x = _sanitize_tensor("After proj_in", x)

    for i, (attn, ff) in enumerate(self.layers):
        latents = attn(x, latents) + latents
        _log_tensor_stats(f"Layer {i} after attn", latents)
        latents = _sanitize_tensor(f"Layer {i} after attn", latents)
        latents = ff(latents) + latents
        _log_tensor_stats(f"Layer {i} after ff", latents)
        latents = _sanitize_tensor(f"Layer {i} after ff", latents)

    latents = self.proj_out(latents)
    _log_tensor_stats("After proj_out", latents)
    latents = _sanitize_tensor("After proj_out", latents)

    output = self.norm_out(latents)
    _log_tensor_stats("After norm_out", output)
    output = _sanitize_tensor("After norm_out", output)
    return output
