"""
SDNQ Integration Layer for Kandinsky5

This module provides SDNQ-based quantization for the Kandinsky5 video generation pipeline.
It replaces the custom INT8 implementation with SDNQ's optimized quantization system,
providing auto-tuned Triton kernels, torch.compile integration, and better performance.

Key features:
- INT8 quantized matmul with auto-tuned kernels
- FP8 quantization support (tensor-wise and row-wise)
- torch.compile integration for speed
- Hardware-aware optimizations (CUDA, ROCm, etc.)
"""

import sys
from pathlib import Path
from typing import Optional
import warnings

import torch
import torch.nn as nn

# Add SDNQ to path
sdnq_path = Path(__file__).parent.parent.parent / "sdnq" / "src"
if str(sdnq_path) not in sys.path:
    sys.path.insert(0, str(sdnq_path))

# Try to import SDNQ
SDNQ_AVAILABLE = False
try:
    from sdnq import SDNQConfig, sdnq_post_load_quant, apply_sdnq_to_module
    from sdnq.common import module_skip_keys_dict, common_skip_keys
    SDNQ_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"SDNQ not available: {e}. Falling back to standard layers.")


# Define skip keys for DiffusionTransformer3D - these layers should NOT be quantized
# because they are critical for numerical stability or too small to benefit
KANDINSKY_SKIP_KEYS = [
    # Time embeddings - critical for temporal conditioning
    "time_embeddings",
    # Text embeddings - projection layers
    "text_embeddings",
    "pooled_text_embeddings",
    # Visual embeddings - patch projection
    "visual_embeddings",
    # RoPE - should stay in original precision
    "text_rope_embeddings",
    "visual_rope_embeddings",
    # Output layer - final projection
    "out_layer",
    # Modulation layers - AdaLN parameters (zero-initialized)
    "text_modulation",
    "visual_modulation",
    # Layer norms (handled by SDNQ automatically but explicit here)
    "self_attention_norm",
    "cross_attention_norm",
    "feed_forward_norm",
    # RMSNorm layers
    "query_norm",
    "key_norm",
]


def register_kandinsky_skip_keys():
    """Register Kandinsky5 model skip keys with SDNQ."""
    if not SDNQ_AVAILABLE:
        return

    # Add DiffusionTransformer3D to the module skip keys dict
    module_skip_keys_dict["DiffusionTransformer3D"] = [
        KANDINSKY_SKIP_KEYS,
        {}  # No dtype overrides needed
    ]


def create_sdnq_linear(
    in_features: int,
    out_features: int,
    bias: bool = True,
    dtype: torch.dtype = torch.bfloat16,
    use_sdnq: bool = True,
) -> nn.Module:
    """
    Factory function to create either SDNQ-ready Linear or standard nn.Linear.

    For SDNQ, we create a standard nn.Linear which will be quantized
    post-loading via sdnq_post_load_quant or apply_sdnq_to_module.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        bias: Whether to use bias
        dtype: Data type for computation
        use_sdnq: Whether SDNQ will be applied (creates standard Linear for now)

    Returns:
        nn.Linear layer ready for SDNQ quantization
    """
    # Always create standard Linear - SDNQ quantizes after loading weights
    return nn.Linear(in_features, out_features, bias=bias, dtype=dtype)


def apply_sdnq_quantization(
    model: nn.Module,
    weights_dtype: str = "int8",
    use_quantized_matmul: bool = True,
    quantized_matmul_dtype: str = None,
    group_size: int = 0,
    use_svd: bool = False,
    svd_rank: int = 32,
    torch_dtype: torch.dtype = torch.bfloat16,
    quantization_device: Optional[torch.device] = None,
    return_device: Optional[torch.device] = None,
    modules_to_skip: list = None,
) -> nn.Module:
    """
    Apply SDNQ quantization to a Kandinsky5 model.

    This should be called AFTER loading weights into the model.

    Args:
        model: The model to quantize (e.g., DiffusionTransformer3D)
        weights_dtype: Target dtype for weights ("int8", "fp8", etc.)
        use_quantized_matmul: Enable quantized INT8/FP8 matmul
        quantized_matmul_dtype: Dtype for matmul ("int8", "fp8", "fp16")
        group_size: Quantization group size (0 = auto)
        use_svd: Enable SVDQuant for better accuracy
        svd_rank: Rank for SVD decomposition
        torch_dtype: Computation dtype
        quantization_device: Device for quantization computation
        return_device: Device to return weights to
        modules_to_skip: Additional modules to skip

    Returns:
        Quantized model
    """
    if not SDNQ_AVAILABLE:
        warnings.warn("SDNQ not available. Model will run without quantization.")
        return model

    # Register Kandinsky skip keys
    register_kandinsky_skip_keys()

    # Merge skip keys
    skip_keys = list(KANDINSKY_SKIP_KEYS)
    if modules_to_skip:
        skip_keys.extend(modules_to_skip)

    print(f"SDNQ: Applying {weights_dtype} quantization...")
    print(f"SDNQ: use_quantized_matmul={use_quantized_matmul}, group_size={group_size}")

    # Apply SDNQ quantization
    model = sdnq_post_load_quant(
        model,
        weights_dtype=weights_dtype,
        quantized_matmul_dtype=quantized_matmul_dtype,
        torch_dtype=torch_dtype,
        group_size=group_size,
        svd_rank=svd_rank,
        svd_steps=8,
        use_svd=use_svd,
        quant_conv=False,  # No convolutions in DiT
        use_quantized_matmul=use_quantized_matmul,
        use_quantized_matmul_conv=False,
        use_stochastic_rounding=False,
        dequantize_fp32=False,
        non_blocking=True,
        add_skip_keys=True,
        quantization_device=quantization_device,
        return_device=return_device,
        modules_to_not_convert=skip_keys,
        modules_dtype_dict={},
    )

    print("SDNQ: Quantization complete")
    return model


class SDNQConfig:
    """Configuration class for SDNQ quantization in Kandinsky5."""

    def __init__(
        self,
        enabled: bool = False,
        weights_dtype: str = "int8",
        use_quantized_matmul: bool = True,
        quantized_matmul_dtype: str = None,
        group_size: int = 0,
        use_svd: bool = False,
        svd_rank: int = 32,
    ):
        """
        Initialize SDNQ configuration.

        Args:
            enabled: Whether to enable SDNQ quantization
            weights_dtype: Target dtype ("int8", "fp8", "int4", etc.)
            use_quantized_matmul: Use accelerated INT8/FP8 matmul
            quantized_matmul_dtype: Matmul dtype (None = auto from weights_dtype)
            group_size: Quantization group size (0 = auto, -1 = per-row)
            use_svd: Enable SVDQuant decomposition
            svd_rank: SVD decomposition rank
        """
        self.enabled = enabled and SDNQ_AVAILABLE
        self.weights_dtype = weights_dtype
        self.use_quantized_matmul = use_quantized_matmul
        self.quantized_matmul_dtype = quantized_matmul_dtype
        self.group_size = group_size
        self.use_svd = use_svd
        self.svd_rank = svd_rank

        if enabled and not SDNQ_AVAILABLE:
            warnings.warn("SDNQ requested but not available. Quantization disabled.")

    def __repr__(self):
        return (
            f"SDNQConfig(enabled={self.enabled}, weights_dtype='{self.weights_dtype}', "
            f"use_quantized_matmul={self.use_quantized_matmul}, group_size={self.group_size})"
        )


def get_sdnq_info() -> dict:
    """Get information about SDNQ availability and capabilities."""
    info = {
        "available": SDNQ_AVAILABLE,
        "version": None,
        "torch_compile": False,
        "triton_mm": False,
        "tensorwise_fp8": True,
    }

    if SDNQ_AVAILABLE:
        try:
            from sdnq.common import sdnq_version, use_torch_compile, use_triton_mm, use_tensorwise_fp8_matmul
            info["version"] = sdnq_version
            info["torch_compile"] = use_torch_compile
            info["triton_mm"] = use_triton_mm
            info["tensorwise_fp8"] = use_tensorwise_fp8_matmul
        except ImportError:
            pass

    return info
