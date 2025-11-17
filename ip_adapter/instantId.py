import torch
from comfy.ldm.modules.attention import optimized_attention

class InstantId(torch.nn.Module):
  def __init__(self, ip_adapter):
    super().__init__()

    self.to_kvs = torch.nn.ModuleDict()

    for key, value in ip_adapter.items():
      k = key.replace(".weight", "").replace(".", "_")
      self.to_kvs[k] = torch.nn.Linear(value.shape[1], value.shape[0], bias=False)
      self.to_kvs[k].weight.data = value


# based on https://github.com/laksjdjf/IPAdapter-ComfyUI/blob/main/ip_adapter.py#L256
class CrossAttentionPatch:
  def __init__(self, scale, instantId, cond, number):
    self.scales = [scale]
    self.instantIds = [instantId]
    self.conds = [cond]
    self.number = number

  def set_new_condition(self, scale, instantId, cond, number):
    self.scales.append(scale)
    self.instantIds.append(instantId)
    self.conds.append(cond)
    self.number = number

  def __call__(self, q, k, v, extra_options):
    # Auto-detect dtype from query tensor to match model precision
    dtype = q.dtype
    print(f"[InstantID Debug] Query dtype: {dtype}, device: {q.device}")

    hidden_states = optimized_attention(q, k, v, extra_options["n_heads"])

    for scale, cond, instantId in zip(self.scales, self.conds,  self.instantIds):
      k_cond = instantId.to_kvs[str(self.number*2+1) + "_to_k_ip"](cond).to(dtype=dtype)
      v_cond = instantId.to_kvs[str(self.number*2+1) + "_to_v_ip"](cond).to(dtype=dtype)

      # Check for NaN/Inf before attention
      if torch.isnan(k_cond).any() or torch.isinf(k_cond).any():
        print(f"[InstantID WARNING] NaN/Inf detected in k_cond at layer {self.number}")
      if torch.isnan(v_cond).any() or torch.isinf(v_cond).any():
        print(f"[InstantID WARNING] NaN/Inf detected in v_cond at layer {self.number}")

      ip_hidden_states = optimized_attention(q, k_cond, v_cond, extra_options["n_heads"])
      hidden_states = hidden_states + ip_hidden_states * scale

      # Check final output
      if torch.isnan(hidden_states).any() or torch.isinf(hidden_states).any():
        print(f"[InstantID ERROR] NaN/Inf in output at layer {self.number}!")
        print(f"  q stats: min={q.min():.4f}, max={q.max():.4f}, mean={q.mean():.4f}")
        print(f"  hidden_states stats: min={hidden_states.min():.4f}, max={hidden_states.max():.4f}")

      return hidden_states.to(dtype=dtype)