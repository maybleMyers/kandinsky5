"""
FP8 Scaled Quantization for Kandinsky Models

This module provides FP8-scaled quantization using PyTorch's native operations
for memory-efficient inference without Triton dependency (works on Windows).

Uses torch.float8_e4m3fn format with per-tensor scaling via torch._scaled_mm.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import warnings


# Check for FP8 support
def check_fp8_support():
    """Check if FP8 is supported on the current hardware."""
    if not torch.cuda.is_available():
        return False, "CUDA not available"

    # FP8 requires compute capability >= 8.9 (Ada Lovelace or newer)
    # or >= 9.0 (Hopper)
    # However, torch._scaled_mm may work on older GPUs with software fallback
    try:
        # Test if float8_e4m3fn dtype is available
        test = torch.zeros(1, dtype=torch.float8_e4m3fn, device='cuda')
        del test
        return True, "FP8 supported"
    except Exception as e:
        return False, f"FP8 not supported: {e}"


# Keys that should NOT be quantized to FP8 (keep in original precision)
FP8_EXCLUDE_KEYS = [
    # Embedding layers - need higher precision
    "pooled_text_embeddings",
    "text_embeddings",
    "visual_embeddings",
    "time_embeddings",

    # Normalization layers (RMSNorm weights)
    "key_norm",
    "query_norm",

    # Patch embedding - initial projection
    "patch_embedding",

    # Head/output layer modulation
    "modulation",

    # Image embedding for I2V
    "img_emb",
]

# Keys that should be targeted for FP8 quantization
FP8_TARGET_KEYS = [
    "blocks",  # transformer blocks contain the linear layers
]


def should_quantize_to_fp8(key: str) -> bool:
    """
    Determine if a tensor should be quantized to FP8.

    Args:
        key: The state dict key name

    Returns:
        True if the tensor should be quantized to FP8
    """
    # Check if it's in an exclude pattern
    for exclude in FP8_EXCLUDE_KEYS:
        if exclude in key:
            return False

    # Check if it's a target for quantization (in blocks)
    for target in FP8_TARGET_KEYS:
        if target in key:
            # Only quantize weight matrices, not biases
            if key.endswith('.weight') and 'norm' not in key:
                return True

    return False


def quantize_tensor_to_fp8(
    tensor: torch.Tensor,
    device: torch.device = None,
    return_on_cpu: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to FP8 format with per-tensor scaling.

    Args:
        tensor: Input tensor to quantize
        device: Device to perform quantization on (for GPU acceleration)
        return_on_cpu: If True, return tensors on CPU to save VRAM

    Returns:
        Tuple of (fp8_tensor, scale)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move to GPU for fast computation
    tensor_gpu = tensor.to(device)

    # Calculate scale factor (per-tensor quantization)
    # FP8 E4M3 has range ~[-448, 448]
    amax = tensor_gpu.abs().max().float()
    scale = amax / 448.0

    # Avoid division by zero
    if scale == 0:
        scale = torch.tensor(1.0, device=device, dtype=torch.float32)
    else:
        scale = scale.to(torch.float32)

    # Scale and convert to FP8
    scaled_tensor = tensor_gpu.float() / scale
    fp8_tensor = scaled_tensor.to(torch.float8_e4m3fn)

    # Free GPU memory immediately
    del tensor_gpu, scaled_tensor

    if return_on_cpu:
        fp8_tensor = fp8_tensor.cpu()
        scale = scale.cpu()

    return fp8_tensor, scale


def optimize_state_dict_with_fp8(
    state_dict: Dict[str, torch.Tensor],
    device: torch.device,
    target_keys: List[str] = None,
    exclude_keys: List[str] = None,
    move_to_device: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Optimize a state dict by converting eligible weights to FP8.

    Processes tensors one at a time to minimize GPU memory usage.

    Args:
        state_dict: Model state dict to optimize
        device: Device to perform optimization on
        target_keys: Keys that must be present for a tensor to be quantized
        exclude_keys: Keys that should never be quantized
        move_to_device: Whether to keep tensors on device after optimization

    Returns:
        Optimized state dict with FP8 weights and scales
    """
    if target_keys is None:
        target_keys = FP8_TARGET_KEYS
    if exclude_keys is None:
        exclude_keys = FP8_EXCLUDE_KEYS

    optimized_dict = {}
    num_quantized = 0
    num_skipped = 0

    for key, tensor in state_dict.items():
        # Check if this tensor should be quantized
        should_quantize = False

        # Must contain a target key
        has_target = any(t in key for t in target_keys)
        # Must not contain an exclude key
        has_exclude = any(e in key for e in exclude_keys)

        if has_target and not has_exclude:
            # Only quantize weight matrices (not biases)
            if key.endswith('.weight') and tensor.ndim == 2:
                should_quantize = True

        if should_quantize:
            # Quantize to FP8 - process on GPU but return on CPU to save VRAM
            # This way we only need GPU memory for one tensor at a time
            fp8_tensor, scale = quantize_tensor_to_fp8(tensor, device, return_on_cpu=True)

            # Store FP8 weight and scale with modified keys
            base_key = key[:-7]  # Remove '.weight'
            optimized_dict[f"{base_key}.weight_fp8"] = fp8_tensor
            optimized_dict[f"{base_key}.weight_scale"] = scale

            num_quantized += 1
        else:
            # Keep original tensor on CPU
            optimized_dict[key] = tensor
            num_skipped += 1

    print(f"FP8 optimization: {num_quantized} weights quantized, {num_skipped} kept in original precision")

    return optimized_dict


class Fp8ScaledLinear(nn.Module):
    """
    FP8 scaled linear layer using PyTorch's native scaled_mm.

    This is a drop-in replacement for nn.Linear that stores weights in FP8
    format and uses torch._scaled_mm for computation.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = dtype

        # Register FP8 weight as buffer
        self.register_buffer(
            'weight_fp8',
            torch.zeros(out_features, in_features, dtype=torch.float8_e4m3fn)
        )

        # Register scale factor
        self.register_buffer(
            'weight_scale',
            torch.ones(1, dtype=torch.float32)
        )

        # Bias in compute dtype
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype))
        else:
            self.register_parameter('bias', None)

        self._is_quantized = False

    def quantize_weights(self, weight: torch.Tensor):
        """Quantize FP32/BF16 weights to FP8 format."""
        device = weight.device if weight.is_cuda else torch.device('cuda')
        fp8_weight, scale = quantize_tensor_to_fp8(weight.to(device), device)

        self.weight_fp8.copy_(fp8_weight)
        self.weight_scale.copy_(scale)
        self._is_quantized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using scaled_mm."""
        if not self._is_quantized:
            raise RuntimeError("Weights not quantized. Call quantize_weights() first.")

        original_shape = x.shape
        original_dtype = x.dtype

        # Reshape to 2D
        x_2d = x.reshape(-1, self.in_features).contiguous()

        # Quantize input to FP8
        x_fp8, x_scale = quantize_tensor_to_fp8(x_2d, x.device)

        # Use scaled_mm: output = (x_fp8 @ weight_fp8.T) * x_scale * weight_scale
        # torch._scaled_mm expects: (M, K) @ (K, N) -> (M, N)
        # Our weight is (out_features, in_features), so we need to transpose
        try:
            # PyTorch's scaled_mm
            out = torch._scaled_mm(
                x_fp8,
                self.weight_fp8.t().contiguous(),
                scale_a=x_scale,
                scale_b=self.weight_scale,
                out_dtype=self.compute_dtype
            )
        except Exception as e:
            # Fallback: dequantize and use regular matmul
            warnings.warn(f"scaled_mm failed: {e}. Using fallback.")
            x_dequant = x_fp8.float() * x_scale
            w_dequant = self.weight_fp8.float() * self.weight_scale
            out = torch.mm(x_dequant, w_dequant.t()).to(self.compute_dtype)

        # Add bias
        if self.bias is not None:
            out = out + self.bias

        # Reshape back
        out = out.reshape(list(original_shape[:-1]) + [self.out_features])

        return out.to(original_dtype)


def apply_fp8_monkey_patch(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    use_scaled_mm: bool = True
):
    """
    Apply FP8 optimization by monkey-patching Linear layers.

    This modifies nn.Linear layers in-place to use FP8 weights and scaled_mm.

    Args:
        model: The model to patch
        state_dict: Optimized state dict with FP8 weights
        use_scaled_mm: Whether to use scaled_mm (if False, uses dequantization)
    """

    def make_fp8_forward(original_forward, weight_fp8, weight_scale, bias, compute_dtype, use_scaled_mm):
        """Create a new forward function that uses FP8 weights."""

        def fp8_forward(x):
            original_shape = x.shape
            in_features = weight_fp8.shape[1]
            out_features = weight_fp8.shape[0]

            # Reshape to 2D
            x_2d = x.reshape(-1, in_features).contiguous()

            if use_scaled_mm:
                # Quantize input
                x_fp8, x_scale = quantize_tensor_to_fp8(x_2d, x.device)

                try:
                    # Use scaled_mm
                    out = torch._scaled_mm(
                        x_fp8,
                        weight_fp8.t().contiguous(),
                        scale_a=x_scale,
                        scale_b=weight_scale,
                        out_dtype=compute_dtype
                    )
                except Exception:
                    # Fallback to dequantization
                    x_dequant = x_fp8.float() * x_scale
                    w_dequant = weight_fp8.float() * weight_scale
                    out = torch.mm(x_dequant, w_dequant.t()).to(compute_dtype)
            else:
                # Dequantize weight and use regular matmul
                w_dequant = weight_fp8.float() * weight_scale
                out = torch.mm(x_2d.float(), w_dequant.t()).to(compute_dtype)

            # Add bias
            if bias is not None:
                out = out + bias

            # Reshape back
            out = out.reshape(list(original_shape[:-1]) + [out_features])

            return out.to(x.dtype)

        return fp8_forward

    # Find all Linear layers and patch them
    patched_count = 0

    def get_module_by_name(model, name):
        """Get a module by its name (dot-separated path)."""
        parts = name.split('.')
        module = model
        for part in parts:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module

    def set_module_by_name(model, name, new_module):
        """Set a module by its name."""
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            if part.isdigit():
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part)

        last_part = parts[-1]
        if last_part.isdigit():
            parent[int(last_part)] = new_module
        else:
            setattr(parent, last_part, new_module)

    # Collect modules to patch
    modules_to_patch = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if we have FP8 weights for this module
            fp8_key = f"{name}.weight_fp8"
            scale_key = f"{name}.weight_scale"

            if fp8_key in state_dict and scale_key in state_dict:
                modules_to_patch.append((name, module, fp8_key, scale_key))

    # Patch the modules
    for name, module, fp8_key, scale_key in modules_to_patch:
        weight_fp8 = state_dict[fp8_key]
        weight_scale = state_dict[scale_key]
        bias = module.bias
        compute_dtype = module.weight.dtype if hasattr(module, 'weight') else torch.bfloat16

        # Create new forward function
        module.forward = make_fp8_forward(
            module.forward,
            weight_fp8,
            weight_scale,
            bias,
            compute_dtype,
            use_scaled_mm
        )

        # Store FP8 data as buffers on the module
        module.register_buffer('weight_fp8', weight_fp8)
        module.register_buffer('weight_scale', weight_scale)

        # Remove original weight to save memory
        if hasattr(module, 'weight'):
            del module.weight

        patched_count += 1

    print(f"FP8 monkey patch applied to {patched_count} Linear layers")

    return patched_count


def fp8_optimization(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    device: torch.device,
    move_to_device: bool = True,
    use_scaled_mm: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Optimize the model with FP8 quantization.

    This is the main entry point for FP8 optimization. It:
    1. Converts eligible weights to FP8 format (one tensor at a time on GPU)
    2. Loads the original weights into model first
    3. Applies monkey patching to use scaled_mm with FP8 weights

    Args:
        model: The model to optimize
        state_dict: The state dict to load
        device: Device for computation (used for FP8 conversion)
        move_to_device: Whether to keep weights on device (usually False, model moved later)
        use_scaled_mm: Whether to use torch._scaled_mm

    Returns:
        Optimized state dict (for compatibility)
    """
    # Check FP8 support
    supported, msg = check_fp8_support()
    if not supported:
        warnings.warn(f"FP8 not fully supported: {msg}. Using software fallback.")

    print(f"FP8 optimization: Converting weights to FP8 format...")

    # First load the original state dict into model (on CPU)
    model.load_state_dict(state_dict, assign=True)

    # Optimize state dict - process tensors one at a time on GPU, keep results on CPU
    optimized_state_dict = optimize_state_dict_with_fp8(
        state_dict,
        device,
        target_keys=FP8_TARGET_KEYS,
        exclude_keys=FP8_EXCLUDE_KEYS,
        move_to_device=False  # Always keep on CPU, model will be moved later
    )

    # Apply monkey patching to replace Linear forwards with FP8 versions
    apply_fp8_monkey_patch(model, optimized_state_dict, use_scaled_mm=use_scaled_mm)

    return optimized_state_dict


def convert_model_to_fp8(
    model: nn.Module,
    device: torch.device = None
) -> int:
    """
    Convert a model's linear layers to FP8 in-place.

    This is an alternative to state dict optimization that converts
    already-loaded weights to FP8.

    Args:
        model: Model to convert
        device: Device for conversion (defaults to model's device)

    Returns:
        Number of layers converted
    """
    if device is None:
        device = next(model.parameters()).device

    converted = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if should quantize
            if should_quantize_to_fp8(name):
                # Get original weight
                weight = module.weight.data
                bias = module.bias

                # Quantize weight
                fp8_weight, scale = quantize_tensor_to_fp8(weight, device)

                # Store as buffers
                module.register_buffer('weight_fp8', fp8_weight)
                module.register_buffer('weight_scale', scale)

                # Create FP8 forward
                compute_dtype = weight.dtype

                def make_forward(w_fp8, w_scale, b, dtype):
                    def forward(x):
                        original_shape = x.shape
                        in_features = w_fp8.shape[1]
                        out_features = w_fp8.shape[0]

                        x_2d = x.reshape(-1, in_features).contiguous()
                        x_fp8, x_scale = quantize_tensor_to_fp8(x_2d, x.device)

                        try:
                            out = torch._scaled_mm(
                                x_fp8, w_fp8.t().contiguous(),
                                scale_a=x_scale, scale_b=w_scale,
                                out_dtype=dtype
                            )
                        except Exception:
                            x_dq = x_fp8.float() * x_scale
                            w_dq = w_fp8.float() * w_scale
                            out = torch.mm(x_dq, w_dq.t()).to(dtype)

                        if b is not None:
                            out = out + b

                        return out.reshape(list(original_shape[:-1]) + [out_features]).to(x.dtype)

                    return forward

                module.forward = make_forward(fp8_weight, scale, bias, compute_dtype)

                # Remove original weight
                del module.weight

                converted += 1

    print(f"Converted {converted} linear layers to FP8")
    return converted
