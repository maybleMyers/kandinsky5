#!/usr/bin/env python3
"""
Convert Kandinsky model checkpoint to INT8 format.

This script pre-converts FP32/BF16 weights to INT8 quantized format,
processing one layer at a time to minimize GPU memory usage.

Usage:
    python scripts/convert_to_int8.py \
        --input_checkpoint path/to/original/model.safetensors \
        --output_checkpoint path/to/output/model_int8.safetensors \
        --config ./configs/config_5s_t2v_pro_20b.yaml \
        --block_size 128
"""

import argparse
import torch
from safetensors.torch import load_file, save_file
from pathlib import Path
import sys
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from RamTorch.kernels.int8_matmul import act_quant
import re

# Layer patterns that should be quantized (from kandinsky/models/int8_utils.py)
INCLUDE_PATTERNS = [
    # FeedForward MLP layers
    r'.*feed_forward\.in_layer.*',
    r'.*feed_forward\.out_layer.*',
    r'.*mlp\.fc1.*',
    r'.*mlp\.fc2.*',

    # Self-attention layers
    r'.*self_attention\.to_query.*',
    r'.*self_attention\.to_key.*',
    r'.*self_attention\.to_value.*',
    r'.*self_attention\.out_layer.*',

    # Cross-attention layers
    r'.*cross_attention\.to_query.*',
    r'.*cross_attention\.to_key.*',
    r'.*cross_attention\.to_value.*',
    r'.*cross_attention\.out_layer.*',
]

# Layer patterns that should NOT be quantized
EXCLUDE_PATTERNS = [
    # Normalization layers
    r'.*norm.*',
    r'.*ln.*',
    r'.*layernorm.*',
    r'.*rmsnorm.*',
    r'.*groupnorm.*',

    # Embedding layers
    r'.*embed.*',
    r'.*embeddings.*',
    r'.*time_embeddings.*',
    r'.*text_embeddings.*',
    r'.*visual_embeddings.*',
    r'.*pooled_text_embeddings.*',
    r'.*rope.*',

    # Modulation layers (adaptive normalization)
    r'.*modulation.*',

    # Output layers (critical for quality)
    r'.*final_layer.*',
]


def convert_weight_to_int8(weight: torch.Tensor, block_size: int = 128):
    """
    Convert a single weight tensor to INT8 format.

    Args:
        weight: Weight tensor [out_features, in_features]
        block_size: Block size for quantization

    Returns:
        Tuple of (weight_int8, weight_scales)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move to GPU and quantize
    weight_gpu = weight.contiguous().to(device)
    weight_int8, weight_scales = act_quant(weight_gpu, block_size=block_size)

    # Move back to CPU to save memory
    weight_int8 = weight_int8.cpu()
    weight_scales = weight_scales.cpu()

    # Clear GPU memory
    del weight_gpu
    torch.cuda.empty_cache()

    return weight_int8, weight_scales


def should_convert_layer(key: str) -> bool:
    """
    Determine if a layer should be converted to INT8.

    Args:
        key: State dict key (e.g., "text_transformer_blocks.0.self_attention.to_query.weight")

    Returns:
        True if layer should be quantized
    """
    # Only convert weight tensors, not biases or other parameters
    if not key.endswith('.weight'):
        return False

    layer_name = key.rsplit('.weight', 1)[0]

    # First check exclude patterns (highest priority)
    for pattern in EXCLUDE_PATTERNS:
        if re.match(pattern, layer_name, re.IGNORECASE):
            return False

    # Then check include patterns
    for pattern in INCLUDE_PATTERNS:
        if re.match(pattern, layer_name, re.IGNORECASE):
            return True

    # Default: don't convert
    return False


def convert_checkpoint_to_int8(
    input_path: str,
    output_path: str,
    block_size: int = 128,
    verbose: bool = True
):
    """
    Convert checkpoint to INT8 format.

    Args:
        input_path: Path to input checkpoint
        output_path: Path to output checkpoint
        block_size: Block size for quantization (must be 128 for Triton)
        verbose: Print progress information
    """
    if verbose:
        print(f"Loading checkpoint from {input_path}")

    # Load original checkpoint
    state_dict = load_file(input_path)

    # Create new state dict for INT8 weights
    int8_state_dict = {}

    # Count layers to convert
    layers_to_convert = [k for k in state_dict.keys() if should_convert_layer(k)]
    total_layers = len(layers_to_convert)

    if verbose:
        print(f"Found {total_layers} layers to convert to INT8")
        print(f"Keeping {len(state_dict) - total_layers} layers in original format")

    # Track statistics
    original_size_mb = 0
    int8_size_mb = 0

    # Process each parameter
    progress_bar = tqdm(state_dict.items(), desc="Converting weights") if verbose else state_dict.items()

    for key, value in progress_bar:
        if should_convert_layer(key):
            # This is a weight tensor to convert
            layer_name = key.rsplit('.weight', 1)[0]

            if verbose and not isinstance(progress_bar, tqdm):
                print(f"Converting {layer_name}")

            # Check dimensions are compatible
            if value.dim() != 2:
                if verbose:
                    print(f"Warning: Skipping {key} - not a 2D tensor (shape: {value.shape})")
                int8_state_dict[key] = value
                continue

            out_features, in_features = value.shape

            if in_features % block_size != 0:
                if verbose:
                    print(f"Warning: Skipping {key} - in_features ({in_features}) not divisible by block_size ({block_size})")
                int8_state_dict[key] = value
                continue

            # Convert to INT8
            try:
                weight_int8, weight_scales = convert_weight_to_int8(value, block_size)

                # Store INT8 weights and scales with new keys
                int8_state_dict[f"{layer_name}.weight_int8"] = weight_int8
                int8_state_dict[f"{layer_name}.weight_scales"] = weight_scales

                # Track size reduction
                original_size_mb += value.element_size() * value.numel() / (1024**2)
                int8_size_mb += (weight_int8.element_size() * weight_int8.numel() +
                                weight_scales.element_size() * weight_scales.numel()) / (1024**2)

            except Exception as e:
                if verbose:
                    print(f"Error converting {key}: {e}")
                    print(f"Keeping original weight")
                int8_state_dict[key] = value
        else:
            # Keep as-is (not a convertible layer, or bias, norm, etc.)
            int8_state_dict[key] = value

    if verbose:
        print(f"\nConversion complete!")
        print(f"Original converted layers size: {original_size_mb:.2f} MB")
        print(f"INT8 converted layers size: {int8_size_mb:.2f} MB")
        print(f"Size reduction: {(1 - int8_size_mb / original_size_mb) * 100:.1f}%")
        print(f"\nSaving INT8 checkpoint to {output_path}")

    # Save INT8 checkpoint
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(int8_state_dict, output_path)

    if verbose:
        print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Convert Kandinsky checkpoint to INT8 format")
    parser.add_argument(
        "--input_checkpoint",
        type=str,
        required=True,
        help="Path to input checkpoint (safetensors format)"
    )
    parser.add_argument(
        "--output_checkpoint",
        type=str,
        required=True,
        help="Path to output INT8 checkpoint"
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=128,
        help="Block size for INT8 quantization (must be 128 for Triton kernels)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    # Validate block size
    if args.block_size != 128:
        print("Warning: Triton kernels require block_size=128. Other values may not work.")

    # Convert checkpoint
    convert_checkpoint_to_int8(
        input_path=args.input_checkpoint,
        output_path=args.output_checkpoint,
        block_size=args.block_size,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
