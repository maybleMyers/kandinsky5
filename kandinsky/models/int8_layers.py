"""
INT8 Quantized Linear Layers for Kandinsky Models

This module provides INT8-quantized linear layers using Triton kernels
for memory-efficient inference with minimal quality loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import warnings

# Import Triton INT8 kernels from RamTorch
try:
    import sys
    from pathlib import Path
    ramtorch_path = Path(__file__).parent.parent.parent / "RamTorch" / "kernels"
    if str(ramtorch_path) not in sys.path:
        sys.path.insert(0, str(ramtorch_path))

    from int8_matmul import act_quant, weight_dequant, int8_gemm
    TRITON_AVAILABLE = True
except ImportError as e:
    TRITON_AVAILABLE = False
    warnings.warn(f"Triton INT8 kernels not available: {e}. INT8 layers will fall back to standard precision.")


class Int8Linear(nn.Module):
    """
    INT8-quantized linear layer using Triton kernels.

    Drop-in replacement for nn.Linear that stores weights in INT8 format
    and performs matrix multiplication using hardware-accelerated INT8 operations.

    Memory savings: 75% compared to FP32, 50% compared to BF16
    Speed: 1.5-2x faster than FP32/BF16 matmul with Triton kernels

    Args:
        in_features: Size of input dimension
        out_features: Size of output dimension
        bias: Whether to include bias term
        block_size: Block size for quantization (must be 128 for Triton)
        dtype: Data type for computation (typically bfloat16)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        block_size: int = 128,
        dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__()

        if not TRITON_AVAILABLE:
            raise RuntimeError(
                "Triton INT8 kernels not available. Please install triton: pip install triton"
            )

        if block_size != 128:
            raise ValueError(
                f"block_size must be 128 for Triton kernels (got {block_size})"
            )

        if in_features % block_size != 0:
            raise ValueError(
                f"in_features ({in_features}) must be divisible by block_size ({block_size})"
            )

        if out_features % block_size != 0:
            raise ValueError(
                f"out_features ({out_features}) must be divisible by block_size ({block_size})"
            )

        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.dtype = dtype

        # Register INT8 weight as buffer (not trainable parameter)
        # Shape: [out_features, in_features]
        self.register_buffer(
            'weight_int8',
            torch.zeros(out_features, in_features, dtype=torch.int8)
        )

        # Register per-block scale factors
        # Block-wise quantization along last dimension only
        # Shape: [out_features, in_features // block_size]
        num_blocks = in_features // block_size
        self.register_buffer(
            'weight_scales',
            torch.ones(out_features, num_blocks, dtype=torch.float32)
        )

        # Bias stays in original dtype
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype))
        else:
            self.register_parameter('bias', None)

        self._is_quantized = False

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """
        Custom state dict loading to handle conversion from FP32 weights to INT8.

        This intercepts the loading process to check if we're loading FP32 weights
        (which have 'weight' key) and converts them to INT8 format.
        Also handles pre-quantized checkpoints that already have INT8 weights.
        """
        weight_key = prefix + 'weight'
        weight_int8_key = prefix + 'weight_int8'
        weight_scales_key = prefix + 'weight_scales'

        # Check if we're loading from a non-INT8 checkpoint (has 'weight' but not 'weight_int8')
        if weight_key in state_dict and weight_int8_key not in state_dict:
            # Load the FP32/BF16 weight
            weight = state_dict[weight_key]

            # Quantize it to INT8
            self.quantize_weights(weight)

            # Remove the 'weight' key from state_dict since we don't have that parameter
            state_dict.pop(weight_key)

            # Remove from missing_keys if they were added
            if weight_int8_key in missing_keys:
                missing_keys.remove(weight_int8_key)
            if weight_scales_key in missing_keys:
                missing_keys.remove(weight_scales_key)
        elif weight_int8_key in state_dict:
            # Loading from pre-quantized checkpoint - mark as quantized
            self._is_quantized = True

        # Call parent's _load_from_state_dict for remaining parameters (like bias)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        block_size: int = 128,
        dtype: torch.dtype = torch.bfloat16
    ) -> 'Int8Linear':
        """
        Create Int8Linear from existing nn.Linear layer.

        Args:
            linear: Source nn.Linear layer
            block_size: Block size for quantization
            dtype: Target dtype for computation

        Returns:
            Int8Linear layer with quantized weights
        """
        int8_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            block_size=block_size,
            dtype=dtype
        )

        # Quantize and store weights
        int8_linear.quantize_weights(linear.weight.data)

        # Copy bias if present
        if linear.bias is not None:
            int8_linear.bias.data.copy_(linear.bias.data.to(dtype))

        return int8_linear

    def quantize_weights(self, weight: torch.Tensor):
        """
        Quantize FP32/BF16 weights to INT8 format.

        Args:
            weight: Weight tensor of shape [out_features, in_features]
        """
        if not TRITON_AVAILABLE:
            raise RuntimeError("Triton kernels not available")

        assert weight.shape == (self.out_features, self.in_features), \
            f"Weight shape mismatch: expected {(self.out_features, self.in_features)}, got {weight.shape}"

        # Ensure weight is contiguous and on GPU (Triton requires CUDA tensors)
        # During load_state_dict, model is still on CPU, so explicitly move to CUDA
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        weight = weight.contiguous().to(device)

        # Quantize using block-wise quantization
        # act_quant expects last dimension to be divisible by block_size
        weight_int8, weight_scales = act_quant(weight, block_size=self.block_size)

        # Move buffers to same device as quantized weights if needed
        if self.weight_int8.device != weight_int8.device:
            self.weight_int8 = self.weight_int8.to(weight_int8.device)
            self.weight_scales = self.weight_scales.to(weight_scales.device)

        # Store quantized weights and scales
        self.weight_int8.copy_(weight_int8)
        self.weight_scales.copy_(weight_scales)
        self._is_quantized = True

    def dequantize_weights(self) -> torch.Tensor:
        """
        Dequantize INT8 weights back to FP32 for inspection/debugging.

        Returns:
            Dequantized weight tensor in FP32
        """
        if not self._is_quantized:
            raise RuntimeError("Weights have not been quantized yet")

        return weight_dequant(
            self.weight_int8,
            self.weight_scales,
            block_size=self.block_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with INT8 quantized matmul.

        Args:
            x: Input tensor of shape [..., in_features]

        Returns:
            Output tensor of shape [..., out_features]
        """
        if not self._is_quantized:
            raise RuntimeError(
                "Weights have not been quantized. Call quantize_weights() first."
            )

        # Store original shape
        original_shape = x.shape
        original_dtype = x.dtype

        # Reshape to 2D for matmul: [..., in_features] -> [batch, in_features]
        x_2d = x.reshape(-1, self.in_features).contiguous()

        # Quantize activations
        # x_2d: [batch, in_features] -> x_int8: [batch, in_features], x_scales: [batch, in_features // block_size]
        x_int8, x_scales = act_quant(x_2d, block_size=self.block_size)

        # INT8 matmul: [batch, in_features] @ [out_features, in_features]^T
        # Result is in FP32
        # Note: weight is stored as [out_features, in_features], so we need to transpose
        # int8_gemm expects: a @ b^T where a is [M, K] and b is [N, K]
        out = int8_gemm(x_int8, x_scales, self.weight_int8, self.weight_scales)

        # Add bias if present
        if self.bias is not None:
            out = out + self.bias

        # Convert to target dtype
        out = out.to(original_dtype)

        # Reshape back to original shape
        output_shape = list(original_shape[:-1]) + [self.out_features]
        out = out.reshape(output_shape)

        return out

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (
            f'in_features={self.in_features}, '
            f'out_features={self.out_features}, '
            f'bias={self.bias is not None}, '
            f'block_size={self.block_size}, '
            f'quantized={self._is_quantized}, '
            f'dtype={self.dtype}'
        )


class Int8LinearFallback(nn.Module):
    """
    Fallback implementation when Triton is not available.

    This is just a wrapper around nn.Linear that provides the same interface
    as Int8Linear but without actual quantization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        block_size: int = 128,
        dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__()

        warnings.warn(
            "Triton not available. Using standard nn.Linear instead of INT8. "
            "Install triton for INT8 acceleration: pip install triton"
        )

        self.linear = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.dtype = dtype
        self._is_quantized = False

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        block_size: int = 128,
        dtype: torch.dtype = torch.bfloat16
    ) -> 'Int8LinearFallback':
        """Create fallback from existing linear layer."""
        fallback = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            block_size=block_size,
            dtype=dtype
        )
        fallback.linear.weight.data.copy_(linear.weight.data.to(dtype))
        if linear.bias is not None:
            fallback.linear.bias.data.copy_(linear.bias.data.to(dtype))
        fallback._is_quantized = True
        return fallback

    def quantize_weights(self, weight: torch.Tensor):
        """Stub for compatibility."""
        self.linear.weight.data.copy_(weight.to(self.dtype))
        self._is_quantized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard linear forward."""
        return self.linear(x)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.linear.bias is not None} (FALLBACK)'


def create_int8_linear(
    in_features: int,
    out_features: int,
    bias: bool = True,
    block_size: int = 128,
    dtype: torch.dtype = torch.bfloat16,
    use_int8: bool = True
) -> nn.Module:
    """
    Factory function to create either Int8Linear or nn.Linear.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        bias: Whether to use bias
        block_size: Block size for INT8 quantization
        dtype: Data type for computation
        use_int8: Whether to use INT8 (if False, returns standard nn.Linear)

    Returns:
        Int8Linear if use_int8=True and Triton available, else nn.Linear
    """
    if use_int8:
        if TRITON_AVAILABLE:
            try:
                return Int8Linear(in_features, out_features, bias, block_size, dtype)
            except (ValueError, RuntimeError) as e:
                warnings.warn(f"Failed to create Int8Linear: {e}. Using standard Linear.")
                return nn.Linear(in_features, out_features, bias=bias, dtype=dtype)
        else:
            return Int8LinearFallback(in_features, out_features, bias, block_size, dtype)
    else:
        return nn.Linear(in_features, out_features, bias=bias, dtype=dtype)
