"""Mixture of Experts forward pass with MXFP4 quantized weights"""
import torch
import triton_kernels.swiglu as swiglu_module
from triton_kernels.matmul_ogs import FnSpecs, FusedActivation, matmul_ogs
from triton_kernels.routing import routing
from triton_kernels.numerics import InFlexData
from triton_kernels.tensor import FP4, convert_layout, wrap_torch_tensor
from triton_kernels.tensor_details import layout
import triton_kernels.matmul_ogs_details.opt_flags as opt_flags

def swizzle_mxfp4(weight_blocks, weight_scales, num_warps=8):
  """Reorder MXFP4 weight layout for H100 tensor cores"""
  value_layout, value_opts = layout.make_default_matmul_mxfp4_w_layout(mx_axis=1)
  scale_layout, scale_opts = layout.make_default_matmul_mxfp4_w_scale_layout(mx_axis=1, num_warps=num_warps)
  opt_flags.update_opt_flags_constraints({"split_k": 1})
  
  weight_blocks = weight_blocks.transpose(-2, -1)
  weight_scales = weight_scales.transpose(-2, -1)
  swizzled_weights = convert_layout(wrap_torch_tensor(weight_blocks, dtype=FP4), value_layout, **value_opts)
  swizzled_scales = convert_layout(wrap_torch_tensor(weight_scales), scale_layout, **scale_opts)
  return swizzled_weights, InFlexData(), swizzled_scales

def moe_forward(hidden_states, gate_up_weights, down_weights, router_logits, num_experts_per_token, renormalize, gate_up_bias=None, down_bias=None, gate_up_precision=None, down_precision=None, apply_weight_on_input=False, inter_cache=None, out_cache=None):
  """Route tokens to top-k experts, apply gated FFN, combine weighted outputs"""
  routing_data, gather_indices, scatter_indices = routing(router_logits, num_experts_per_token, sm_first=not renormalize)
  
  num_tokens, hidden_dim = hidden_states.shape
  num_experts, _, intermediate_dim = gate_up_weights.shape
  
  if gate_up_bias is not None and gate_up_bias.dtype != torch.float32: gate_up_bias = gate_up_bias.float()
  if down_bias is not None and down_bias.dtype != torch.float32: down_bias = down_bias.float()
  
  intermediate_size = intermediate_dim // 2
  if inter_cache is not None and inter_cache.numel() >= num_tokens * num_experts_per_token * intermediate_size:
    intermediate = inter_cache.view(-1)[:num_tokens * num_experts_per_token * intermediate_size].view(1, num_tokens * num_experts_per_token, intermediate_size)
  else:
    intermediate = torch.empty((1, num_tokens * num_experts_per_token, intermediate_size), device=hidden_states.device, dtype=hidden_states.dtype)
  
  if out_cache is not None and out_cache.numel() >= num_tokens * hidden_dim:
    output = out_cache.view(-1)[:num_tokens * hidden_dim].view(1, num_tokens, hidden_dim)
  else:
    output = torch.empty((1, num_tokens, hidden_dim), device=hidden_states.device, dtype=hidden_states.dtype)
  
  swiglu_activation = FusedActivation(FnSpecs("swiglu", swiglu_module.swiglu_fn, ("alpha", "limit")), (1.702, 7.0), 2)
  expert_weights = routing_data.gate_scal if routing_data else None
  
  matmul_ogs(hidden_states, gate_up_weights, gate_up_bias, routing_data, gather_indx=gather_indices, precision_config=gate_up_precision, gammas=expert_weights if apply_weight_on_input else None, fused_activation=swiglu_activation, y=intermediate)
  matmul_ogs(intermediate.view(num_tokens * num_experts_per_token, intermediate_size), down_weights, down_bias, routing_data, scatter_indx=scatter_indices, precision_config=down_precision, gammas=None if apply_weight_on_input else expert_weights, y=output)
  
  return output.view(num_tokens, hidden_dim)
