# modified from https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/src/helpers.py
import math

import torch
import torch.nn as nn

# FFN
def FeedForward(dim, mult=4):
  inner_dim = int(dim * mult)
  return nn.Sequential(
    nn.LayerNorm(dim),
    nn.Linear(dim, inner_dim, bias=False),
    nn.GELU(),
    nn.Linear(inner_dim, dim, bias=False),
  )

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
    print(f"[Resampler] Input x: min={x.min().item():.2f}, max={x.max().item():.2f}, mean={x.mean().item():.2f}, has_nan={torch.isnan(x).any().item()}")

    latents = self.latents.to(x.device).repeat(x.size(0), 1, 1)
    print(f"[Resampler] Latents: min={latents.min().item():.2f}, max={latents.max().item():.2f}, has_nan={torch.isnan(latents).any().item()}")

    x = self.proj_in(x)
    print(f"[Resampler] After proj_in: min={x.min().item():.2f}, max={x.max().item():.2f}, has_nan={torch.isnan(x).any().item()}")

    for i, (attn, ff) in enumerate(self.layers):
        latents = attn(x, latents) + latents
        print(f"[Resampler] Layer {i} after attn: min={latents.min().item():.2f}, max={latents.max().item():.2f}, has_nan={torch.isnan(latents).any().item()}")
        latents = ff(latents) + latents
        print(f"[Resampler] Layer {i} after ff: min={latents.min().item():.2f}, max={latents.max().item():.2f}, has_nan={torch.isnan(latents).any().item()}")

    latents = self.proj_out(latents)
    print(f"[Resampler] After proj_out: min={latents.min().item():.2f}, max={latents.max().item():.2f}, mean={latents.mean().item():.2f}")
    output = self.norm_out(latents)
    print(f"[Resampler] After norm_out: min={output.min().item():.2f}, max={output.max().item():.2f}, mean={output.mean().item():.2f}")
    return output