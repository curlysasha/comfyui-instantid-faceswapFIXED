# modified from https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/src/helpers.py
import math

import torch
import torch.nn as nn

# FFN with gradient clipping to prevent overflow
class FeedForward(nn.Module):
  def __init__(self, dim, mult=4):
    super().__init__()
    inner_dim = int(dim * mult)
    self.norm = nn.LayerNorm(dim)
    self.fc1 = nn.Linear(dim, inner_dim, bias=False)
    self.act = nn.GELU()
    self.fc2 = nn.Linear(inner_dim, dim, bias=False)

  def forward(self, x):
    x = self.norm(x)
    x = torch.clamp(x, min=-10, max=10)  # Prevent overflow
    x = self.fc1(x)
    x = torch.clamp(x, min=-10, max=10)  # Prevent overflow
    x = self.act(x)
    x = self.fc2(x)
    x = torch.clamp(x, min=-10, max=10)  # Prevent overflow
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

    # Clamp q and k to prevent overflow in matmul
    q = torch.clamp(q, min=-10, max=10)
    k = torch.clamp(k, min=-10, max=10)

    weight = (q * scale) @ (k * scale).transpose(-2, -1) # More stable with f16 than dividing afterwards

    # Clamp weight before softmax to prevent overflow
    weight = torch.clamp(weight, min=-50, max=50)

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
    print(f"[Resampler Debug] Input dtype: {x.dtype}, shape: {x.shape}, device: {x.device}")

    # Check input for NaN/Inf
    if torch.isnan(x).any() or torch.isinf(x).any():
      print(f"[Resampler ERROR] Input already has NaN/Inf!")
      print(f"  Input stats: min={x.min():.4f}, max={x.max():.4f}, mean={x.mean():.4f}")
      return torch.zeros(x.size(0), 16, 2048, dtype=x.dtype, device=x.device)

    # Move latents to same device as input
    latents = self.latents.to(x.device, dtype=x.dtype).repeat(x.size(0), 1, 1)
    print(f"[Resampler] Latents stats: min={latents.min():.4f}, max={latents.max():.4f}, device={latents.device}, dtype={latents.dtype}")

    x = self.proj_in(x)
    print(f"[Resampler] After proj_in: has_nan={torch.isnan(x).any()}, min={x.min():.4f}, max={x.max():.4f}")

    if torch.isnan(x).any() or torch.isinf(x).any():
      print(f"[Resampler ERROR] NaN/Inf after proj_in!")
      return torch.zeros(x.size(0), 16, 2048, dtype=x.dtype, device=x.device)

    for i, (attn, ff) in enumerate(self.layers):
        latents_before = latents
        latents = attn(x, latents) + latents

        if torch.isnan(latents).any() or torch.isinf(latents).any():
          print(f"[Resampler ERROR] NaN/Inf after attention layer {i}!")
          print(f"  latents_before stats: min={latents_before.min():.4f}, max={latents_before.max():.4f}")
          return torch.zeros(x.size(0), 16, 2048, dtype=x.dtype, device=x.device)

        latents = ff(latents) + latents

        if torch.isnan(latents).any() or torch.isinf(latents).any():
          print(f"[Resampler ERROR] NaN/Inf after feedforward layer {i}!")
          return torch.zeros(x.size(0), 16, 2048, dtype=x.dtype, device=x.device)

    latents = self.proj_out(latents)
    print(f"[Resampler] After proj_out: has_nan={torch.isnan(latents).any()}, min={latents.min():.4f}, max={latents.max():.4f}")

    output = self.norm_out(latents)

    # Check for NaN/Inf before clamping
    has_nan = torch.isnan(output).any()
    has_inf = torch.isinf(output).any()
    if has_nan or has_inf:
      print(f"[Resampler WARNING] Pre-clamp NaN: {has_nan}, Inf: {has_inf}")
      print(f"  Output stats: min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")
      # Return zeros instead of NaN
      print(f"[Resampler] Returning zero tensor to prevent NaN propagation")
      return torch.zeros_like(output)

    # Clamp output to prevent NaN/Inf propagation
    output = torch.clamp(output, min=-65504, max=65504)

    # Final check
    if torch.isnan(output).any() or torch.isinf(output).any():
      print(f"[Resampler ERROR] NaN/Inf still present after clamping!")
      return torch.zeros_like(output)

    return output