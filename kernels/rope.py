"""Rotary Position Embedding Triton kernel"""
import torch, triton, triton.language as tl

@triton.jit
def _rope_decode_kernel(query_ptr, key_ptr, cos_ptr, sin_ptr, positions_ptr, query_out_ptr, key_out_ptr, batch_size, num_query_heads, num_key_heads, head_dim: tl.constexpr, query_batch_stride, query_head_stride, query_dim_stride, key_batch_stride, key_head_stride, key_dim_stride, cos_pos_stride, cos_dim_stride, BLOCK: tl.constexpr):
  batch_idx = tl.program_id(0)
  head_idx = tl.program_id(1)
  if batch_idx >= batch_size: return
  
  position = tl.load(positions_ptr + batch_idx)
  dim_offsets = tl.arange(0, head_dim)
  cos_vals = tl.load(cos_ptr + position * cos_pos_stride + dim_offsets * cos_dim_stride)
  sin_vals = tl.load(sin_ptr + position * cos_pos_stride + dim_offsets * cos_dim_stride)
  half_dim: tl.constexpr = head_dim // 2
  
  if head_idx < num_query_heads:
    query_base = query_ptr + batch_idx * query_batch_stride + head_idx * query_head_stride
    query_vals = tl.load(query_base + dim_offsets * query_dim_stride)
    rotated_idx = tl.where(dim_offsets < half_dim, dim_offsets + half_dim, dim_offsets - half_dim)
    rotated_vals = tl.load(query_base + rotated_idx * query_dim_stride)
    rotated_query = tl.where(dim_offsets < half_dim, -rotated_vals, rotated_vals)
    query_rotated = query_vals * cos_vals + rotated_query * sin_vals
    tl.store(query_out_ptr + batch_idx * query_batch_stride + head_idx * query_head_stride + dim_offsets * query_dim_stride, query_rotated)
  
  if head_idx < num_key_heads:
    key_base = key_ptr + batch_idx * key_batch_stride + head_idx * key_head_stride
    key_vals = tl.load(key_base + dim_offsets * key_dim_stride)
    rotated_idx = tl.where(dim_offsets < half_dim, dim_offsets + half_dim, dim_offsets - half_dim)
    rotated_vals = tl.load(key_base + rotated_idx * key_dim_stride)
    rotated_key = tl.where(dim_offsets < half_dim, -rotated_vals, rotated_vals)
    key_rotated = key_vals * cos_vals + rotated_key * sin_vals
    tl.store(key_out_ptr + batch_idx * key_batch_stride + head_idx * key_head_stride + dim_offsets * key_dim_stride, key_rotated)

def fused_rope(positions, query, key, cos_cache, sin_cache):
  batch_size, seq_len, num_query_heads, head_dim = query.shape
  num_key_heads = key.shape[2]
  query_out, key_out = torch.empty_like(query), torch.empty_like(key)
  
  if not query.is_contiguous(): query = query.contiguous()
  if not key.is_contiguous(): key = key.contiguous()
  
  if seq_len == 1:
    positions_flat = positions.view(-1) if positions.dim() > 1 else positions
    query_squeezed = query.squeeze(1)
    key_squeezed = key.squeeze(1)
    query_out_squeezed = query_out.squeeze(1)
    key_out_squeezed = key_out.squeeze(1)
    grid = (batch_size, max(num_query_heads, num_key_heads))
    _rope_decode_kernel[grid](query_squeezed, key_squeezed, cos_cache, sin_cache, positions_flat, query_out_squeezed, key_out_squeezed, batch_size, num_query_heads, num_key_heads, head_dim, query_squeezed.stride(0), query_squeezed.stride(1), query_squeezed.stride(2), key_squeezed.stride(0), key_squeezed.stride(1), key_squeezed.stride(2), cos_cache.stride(0), cos_cache.stride(1), BLOCK=1)
  else:
    cos_vals = cos_cache[positions]
    sin_vals = sin_cache[positions]
    if positions.dim() == 1:
      cos_vals = cos_vals.unsqueeze(0).unsqueeze(2)
      sin_vals = sin_vals.unsqueeze(0).unsqueeze(2)
    else:
      cos_vals = cos_vals.unsqueeze(2)
      sin_vals = sin_vals.unsqueeze(2)
    half_dim = head_dim // 2
    query_out = query * cos_vals + torch.cat((-query[..., half_dim:], query[..., :half_dim]), -1) * sin_vals
    key_out = key * cos_vals + torch.cat((-key[..., half_dim:], key[..., :half_dim]), -1) * sin_vals
  
  return query_out, key_out
