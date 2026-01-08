"""RMSNorm Triton kernels"""
import torch, triton, triton.language as tl

@triton.jit
def _rms_norm_kernel(input_ptr, weight_ptr, output_ptr, input_stride, output_stride, hidden_size, eps: tl.constexpr, BLOCK: tl.constexpr):
  row_idx = tl.program_id(0)
  col_offsets = tl.arange(0, BLOCK)
  mask = col_offsets < hidden_size
  
  input_row = tl.load(input_ptr + row_idx * input_stride + col_offsets, mask=mask, other=0.0).to(tl.float32)
  weight = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
  
  variance = tl.sum(input_row * input_row, axis=0) / hidden_size
  normalized = input_row * tl.rsqrt(variance + eps) * weight
  tl.store(output_ptr + row_idx * output_stride + col_offsets, normalized.to(tl.bfloat16), mask=mask)

@triton.jit
def _fused_add_rms_norm_kernel(input_ptr, residual_ptr, weight_ptr, output_ptr, residual_out_ptr, input_stride, residual_stride, output_stride, residual_out_stride, hidden_size, eps: tl.constexpr, BLOCK: tl.constexpr):
  row_idx = tl.program_id(0)
  col_offsets = tl.arange(0, BLOCK)
  mask = col_offsets < hidden_size
  
  input_row = tl.load(input_ptr + row_idx * input_stride + col_offsets, mask=mask, other=0.0).to(tl.float32)
  residual_row = tl.load(residual_ptr + row_idx * residual_stride + col_offsets, mask=mask, other=0.0).to(tl.float32)
  weight = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
  
  hidden = input_row + residual_row
  tl.store(residual_out_ptr + row_idx * residual_out_stride + col_offsets, hidden.to(tl.bfloat16), mask=mask)
  
  variance = tl.sum(hidden * hidden, axis=0) / hidden_size
  normalized = hidden * tl.rsqrt(variance + eps) * weight
  tl.store(output_ptr + row_idx * output_stride + col_offsets, normalized.to(tl.bfloat16), mask=mask)

def rms_norm(input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
  original_shape = input.shape
  input_2d = input.view(-1, original_shape[-1])
  num_rows, hidden_size = input_2d.shape
  output = torch.empty_like(input_2d)
  block_size = min(triton.next_power_of_2(hidden_size), 8192)
  _rms_norm_kernel[(num_rows,)](input_2d, weight, output, input_2d.stride(0), output.stride(0), hidden_size, eps, block_size)
  return output.view(original_shape)

def fused_add_rms_norm(input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5):
  original_shape = input.shape
  input_2d = input.view(-1, original_shape[-1])
  residual_2d = residual.view(-1, original_shape[-1])
  num_rows, hidden_size = input_2d.shape
  output = torch.empty_like(input_2d)
  residual_out = torch.empty_like(residual_2d)
  block_size = min(triton.next_power_of_2(hidden_size), 8192)
  _fused_add_rms_norm_kernel[(num_rows,)](input_2d, residual_2d, weight, output, residual_out, input_2d.stride(0), residual_2d.stride(0), output.stride(0), residual_out.stride(0), hidden_size, eps, block_size)
  return output.view(original_shape), residual_out.view(original_shape)
