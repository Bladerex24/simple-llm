"""Custom Triton kernels for optimized inference"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))) if os.path.dirname(os.path.abspath(__file__)) not in sys.path else None

from .norm import rms_norm, fused_add_rms_norm
from .rope import fused_rope
from .moe import moe_forward, swizzle_mxfp4
