"""
Utilities for INT8 quantization in Kandinsky models.

Functions to determine which layers should be quantized and helper
utilities for managing INT8 quantization configuration.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Set
import re


class Int8Config:
    """Configuration for INT8 quantization."""

    def __init__(
        self,
        enabled: bool = True,
        block_size: int = 128,
        min_layer_size: int = 4096,
        strategy: str = "auto",
        quantize_feedforward: bool = True,
        quantize_attention: bool = True,
        quantize_embeddings: bool = False,
        quantize_output: bool = False,
    ):
        """
        Args:
            enabled: Whether INT8 quantization is enabled
            block_size: Block size for quantization (must be 128 for Triton)
            min_layer_size: Minimum number of parameters to quantize a layer
            strategy: Quantization strategy - "auto", "aggressive", "conservative"
            quantize_feedforward: Whether to quantize FeedForward layers
            quantize_attention: Whether to quantize attention QKV and projection layers
            quantize_embeddings: Whether to quantize embedding layers (not recommended)
            quantize_output: Whether to quantize output layers (not recommended)
        """
        self.enabled = enabled
        self.block_size = block_size
        self.min_layer_size = min_layer_size
        self.strategy = strategy
        self.quantize_feedforward = quantize_feedforward
        self.quantize_attention = quantize_attention
        self.quantize_embeddings = quantize_embeddings
        self.quantize_output = quantize_output

        # Adjust settings based on strategy
        if strategy == "aggressive":
            self.min_layer_size = 2048
            self.quantize_feedforward = True
            self.quantize_attention = True
        elif strategy == "conservative":
            self.min_layer_size = 8192
            self.quantize_feedforward = True
            self.quantize_attention = False  # Only feedforward, safer
        # "auto" uses defaults

    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            "enabled": self.enabled,
            "block_size": self.block_size,
            "min_layer_size": self.min_layer_size,
            "strategy": self.strategy,
            "quantize_feedforward": self.quantize_feedforward,
            "quantize_attention": self.quantize_attention,
            "quantize_embeddings": self.quantize_embeddings,
            "quantize_output": self.quantize_output,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'Int8Config':
        """Create config from dictionary."""
        return cls(**config_dict)


# Layer patterns that should NOT be quantized (critical for quality)
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
    r'.*out_layer.*',
    r'.*final_layer.*',
]

# Layer patterns that are good candidates for INT8
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


def should_quantize_layer(
    layer_name: str,
    layer: nn.Module,
    config: Int8Config
) -> bool:
    """
    Determine if a layer should be quantized to INT8.

    Args:
        layer_name: Full name/path of the layer
        layer: The layer module
        config: INT8 configuration

    Returns:
        True if layer should be quantized, False otherwise
    """
    if not config.enabled:
        return False

    # Only quantize nn.Linear layers
    if not isinstance(layer, nn.Linear):
        return False

    # Check size threshold
    num_params = layer.in_features * layer.out_features
    if num_params < config.min_layer_size:
        return False

    # Check dimensions are divisible by block size
    if layer.in_features % config.block_size != 0:
        return False
    if layer.out_features % config.block_size != 0:
        return False

    layer_name_lower = layer_name.lower()

    # First check exclude patterns (these take priority)
    for pattern in EXCLUDE_PATTERNS:
        if re.match(pattern, layer_name_lower):
            return False

    # Check if embeddings are allowed
    if not config.quantize_embeddings:
        if 'embed' in layer_name_lower:
            return False

    # Check if output layers are allowed
    if not config.quantize_output:
        if 'out_layer' in layer_name_lower or 'final_layer' in layer_name_lower:
            return False

    # Check include patterns based on config
    if config.quantize_feedforward:
        if any(re.match(p, layer_name_lower) for p in INCLUDE_PATTERNS if 'feed_forward' in p or 'mlp' in p):
            return True

    if config.quantize_attention:
        if any(re.match(p, layer_name_lower) for p in INCLUDE_PATTERNS if 'attention' in p):
            return True

    # Default: don't quantize if not explicitly matched
    return False


def get_quantizable_layers(
    model: nn.Module,
    config: Int8Config,
    prefix: str = ""
) -> Dict[str, nn.Linear]:
    """
    Get all layers in the model that should be quantized.

    Args:
        model: PyTorch model
        config: INT8 configuration
        prefix: Prefix for layer names (for recursion)

    Returns:
        Dictionary mapping layer names to nn.Linear modules
    """
    quantizable = {}

    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name

        if isinstance(module, nn.Linear):
            if should_quantize_layer(full_name, module, config):
                quantizable[full_name] = module
        else:
            # Recursively check child modules
            child_quantizable = get_quantizable_layers(module, config, full_name)
            quantizable.update(child_quantizable)

    return quantizable


def analyze_model_for_int8(
    model: nn.Module,
    config: Int8Config
) -> Dict:
    """
    Analyze a model to estimate INT8 quantization impact.

    Args:
        model: PyTorch model to analyze
        config: INT8 configuration

    Returns:
        Dictionary with analysis results including:
        - total_params: Total number of parameters
        - quantizable_params: Number of parameters that will be quantized
        - memory_original_gb: Original memory usage in GB
        - memory_int8_gb: Memory usage with INT8 in GB
        - memory_savings_gb: Memory saved in GB
        - memory_savings_pct: Percentage of memory saved
        - quantizable_layers: List of layer names that will be quantized
    """
    total_params = sum(p.numel() for p in model.parameters())
    quantizable_layers = get_quantizable_layers(model, config)
    quantizable_params = sum(
        layer.in_features * layer.out_features
        for layer in quantizable_layers.values()
    )
    non_quantizable_params = total_params - quantizable_params

    # Memory calculations (assuming FP32 original, BF16 for non-quantized)
    memory_original_gb = (total_params * 4) / (1024**3)  # FP32
    memory_int8_gb = (
        (quantizable_params * 1) +  # INT8 weights
        (non_quantizable_params * 2)  # BF16 for others
    ) / (1024**3)
    memory_savings_gb = memory_original_gb - memory_int8_gb
    memory_savings_pct = (memory_savings_gb / memory_original_gb) * 100 if memory_original_gb > 0 else 0

    return {
        "total_params": total_params,
        "quantizable_params": quantizable_params,
        "non_quantizable_params": non_quantizable_params,
        "quantizable_pct": (quantizable_params / total_params * 100) if total_params > 0 else 0,
        "memory_original_gb": round(memory_original_gb, 2),
        "memory_int8_gb": round(memory_int8_gb, 2),
        "memory_savings_gb": round(memory_savings_gb, 2),
        "memory_savings_pct": round(memory_savings_pct, 1),
        "num_quantizable_layers": len(quantizable_layers),
        "quantizable_layers": list(quantizable_layers.keys()),
    }


def print_int8_analysis(analysis: Dict):
    """
    Pretty print INT8 analysis results.

    Args:
        analysis: Analysis dictionary from analyze_model_for_int8
    """
    print("\n" + "=" * 80)
    print("INT8 QUANTIZATION ANALYSIS")
    print("=" * 80)

    print(f"\nParameters:")
    print(f"  Total: {analysis['total_params']:,}")
    print(f"  Quantizable to INT8: {analysis['quantizable_params']:,} ({analysis['quantizable_pct']:.1f}%)")
    print(f"  Remaining in BF16/FP32: {analysis['non_quantizable_params']:,}")

    print(f"\nMemory Usage:")
    print(f"  Original (FP32): {analysis['memory_original_gb']:.2f} GB")
    print(f"  With INT8: {analysis['memory_int8_gb']:.2f} GB")
    print(f"  Savings: {analysis['memory_savings_gb']:.2f} GB ({analysis['memory_savings_pct']:.1f}%)")

    print(f"\nQuantizable Layers: {analysis['num_quantizable_layers']}")
    if analysis['num_quantizable_layers'] > 0:
        print(f"  First 10 layers:")
        for i, layer_name in enumerate(analysis['quantizable_layers'][:10]):
            print(f"    {i+1}. {layer_name}")
        if analysis['num_quantizable_layers'] > 10:
            print(f"    ... and {analysis['num_quantizable_layers'] - 10} more")

    print("=" * 80)


def count_linear_parameters(module: nn.Module) -> Tuple[int, int]:
    """
    Count parameters in linear layers vs total.

    Args:
        module: PyTorch module

    Returns:
        Tuple of (linear_params, total_params)
    """
    total_params = sum(p.numel() for p in module.parameters())
    linear_params = 0

    for m in module.modules():
        if isinstance(m, nn.Linear):
            linear_params += sum(p.numel() for p in m.parameters())

    return linear_params, total_params


def check_layer_compatibility(layer: nn.Linear, block_size: int = 128) -> Tuple[bool, str]:
    """
    Check if a layer is compatible with INT8 quantization.

    Args:
        layer: Linear layer to check
        block_size: Block size for quantization

    Returns:
        Tuple of (is_compatible, reason)
    """
    if not isinstance(layer, nn.Linear):
        return False, "Not a Linear layer"

    if layer.in_features % block_size != 0:
        return False, f"in_features ({layer.in_features}) not divisible by block_size ({block_size})"

    if layer.out_features % block_size != 0:
        return False, f"out_features ({layer.out_features}) not divisible by block_size ({block_size})"

    num_params = layer.in_features * layer.out_features
    if num_params < 4096:
        return False, f"Layer too small ({num_params} params < 4096 minimum)"

    return True, "Compatible"
