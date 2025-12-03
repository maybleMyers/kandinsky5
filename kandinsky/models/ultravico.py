"""
UltraViCo: Ultra-extrapolated Video via Attention Concentration

Based on the paper: "UltraViCo: Breaking Extrapolation Limits in Video Diffusion Transformers"
https://arxiv.org/abs/2511.20123

This module implements attention decay for video length extrapolation, addressing:
1. Quality degradation (universal) - caused by attention dispersion
2. Content repetition (model-specific) - caused by harmonic RoPE frequencies

The key insight is that tokens beyond the training window dilute learned attention patterns.
By applying a decay factor to out-of-window attention, we restore focus on reliable context.

USAGE:
    This is ONLY activated when --ultravico flag is passed to test.py
    Existing code paths remain completely unchanged without the flag.
"""

import math
import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class UltraViCoConfig:
    """Configuration for UltraViCo attention decay."""
    enabled: bool = False
    # Training window size in latent frames (5s=31, 10s=61 for Kandinsky5)
    training_frames: int = 31
    # Decay factor for out-of-window attention (0.85-0.95 recommended)
    alpha: float = 0.9
    # Decay factor for harmonic risk positions (only if repetition occurs)
    beta: float = 0.6
    # Whether to apply harmonic suppression (set True if you see repetition)
    suppress_harmonics: bool = False
    # Frames around harmonic peaks to suppress
    gamma: int = 4
    # Harmonic period in latent frames (auto-detected or manual)
    harmonic_period: Optional[int] = None


def compute_temporal_positions(seq_len: int, height: int, width: int, device: torch.device) -> torch.Tensor:
    """
    Compute temporal position for each token in flattened sequence.

    Tokens are flattened in (T, H, W) order, so:
    - token i has temporal position: t_i = i // (H * W)

    Args:
        seq_len: Total sequence length (T * H * W)
        height: Spatial height in latent space
        width: Spatial width in latent space
        device: Device for tensor

    Returns:
        Tensor of shape [seq_len] with temporal position for each token
    """
    hw = height * width
    temporal_pos = torch.arange(seq_len, device=device) // hw
    return temporal_pos


def create_ultravico_bias(
    seq_len: int,
    height: int,
    width: int,
    config: UltraViCoConfig,
    device: torch.device,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Create attention bias for UltraViCo decay.

    The bias is added to attention logits before softmax:
    - 0 for in-window pairs (no change)
    - log(alpha) for out-of-window pairs (multiplicative decay after softmax)
    - log(beta) for harmonic risk positions (stronger decay)

    Args:
        seq_len: Total sequence length
        height: Spatial height in latent space
        width: Spatial width in latent space
        config: UltraViCo configuration
        device: Device for tensor
        dtype: Data type for tensor

    Returns:
        Attention bias tensor of shape [seq_len, seq_len]
    """
    # Get temporal positions
    t_pos = compute_temporal_positions(seq_len, height, width, device)

    # Compute temporal distance matrix
    # t_dist[i, j] = |t_i - t_j|
    t_dist = torch.abs(t_pos.unsqueeze(1) - t_pos.unsqueeze(0))

    # Training window radius (half of training frames)
    window_radius = config.training_frames // 2

    # Create bias tensor (0 = no change, negative = decay)
    bias = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)

    # Apply alpha decay to out-of-window positions
    out_of_window = t_dist > window_radius
    if config.alpha < 1.0:
        # log(alpha) as additive bias = multiplicative alpha after exp
        bias[out_of_window] = math.log(config.alpha)

    # Apply beta decay to harmonic risk positions (if enabled)
    if config.suppress_harmonics and config.beta < config.alpha:
        period = config.harmonic_period
        if period is None:
            # Auto-detect: for Kandinsky5 with axes_dims[0]=16, period ≈ training_frames
            period = config.training_frames

        # Find positions near harmonic alignment: |t_i - t_j| ≈ m * period
        for m in range(1, (t_dist.max().item() // period) + 2):
            harmonic_center = m * period
            near_harmonic = (t_dist >= harmonic_center - config.gamma) & \
                           (t_dist <= harmonic_center + config.gamma)
            # Only apply to out-of-window positions
            risk_positions = near_harmonic & out_of_window
            bias[risk_positions] = math.log(config.beta)

    return bias


def create_ultravico_bias_for_shape(
    shape: Tuple[int, int, int],
    config: UltraViCoConfig,
    device: torch.device,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Create UltraViCo bias given visual shape (T, H, W).

    Args:
        shape: Visual shape tuple (duration, height, width) in latent space
        config: UltraViCo configuration
        device: Device for tensor
        dtype: Data type

    Returns:
        Attention bias tensor
    """
    duration, height, width = shape
    seq_len = duration * height * width
    return create_ultravico_bias(seq_len, height, width, config, device, dtype)


def apply_ultravico_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    ultravico_bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Apply scaled dot-product attention with UltraViCo bias.

    This is a drop-in replacement for standard attention that applies
    the UltraViCo decay via attention bias.

    Args:
        query: Query tensor [batch, seq, heads, dim] or [seq, heads, dim]
        key: Key tensor
        value: Value tensor
        ultravico_bias: Optional bias tensor [seq, seq]

    Returns:
        Attention output tensor
    """
    # Add batch dim if needed
    squeeze_batch = False
    if query.dim() == 3:
        query = query.unsqueeze(0)
        key = key.unsqueeze(0)
        value = value.unsqueeze(0)
        squeeze_batch = True

    # Transpose for SDPA: [batch, heads, seq, dim]
    query = query.transpose(1, 2).contiguous()
    key = key.transpose(1, 2).contiguous()
    value = value.transpose(1, 2).contiguous()

    # Apply attention with bias
    if ultravico_bias is not None:
        # Expand bias for batch and heads: [1, 1, seq, seq]
        bias = ultravico_bias.unsqueeze(0).unsqueeze(0)
        out = F.scaled_dot_product_attention(query, key, value, attn_mask=bias)
    else:
        out = F.scaled_dot_product_attention(query, key, value)

    # Transpose back: [batch, seq, heads, dim]
    out = out.transpose(1, 2).contiguous()

    if squeeze_batch:
        out = out.squeeze(0)

    return out


# Global config instance (can be set from CLI)
_ultravico_config: Optional[UltraViCoConfig] = None
_cached_bias: Optional[torch.Tensor] = None
_cached_shape: Optional[Tuple[int, int, int]] = None
_current_visual_shape: Optional[Tuple[int, int, int]] = None


def set_ultravico_config(config: UltraViCoConfig):
    """Set global UltraViCo configuration."""
    global _ultravico_config, _cached_bias, _cached_shape
    _ultravico_config = config
    _cached_bias = None
    _cached_shape = None


def get_ultravico_config() -> Optional[UltraViCoConfig]:
    """Get global UltraViCo configuration."""
    return _ultravico_config


def is_ultravico_enabled() -> bool:
    """Check if UltraViCo is enabled."""
    return _ultravico_config is not None and _ultravico_config.enabled


def set_current_visual_shape(shape: Tuple[int, int, int]):
    """Set current visual shape for attention bias computation."""
    global _current_visual_shape
    _current_visual_shape = shape


def get_current_visual_shape() -> Optional[Tuple[int, int, int]]:
    """Get current visual shape."""
    return _current_visual_shape


def get_ultravico_bias(shape: Tuple[int, int, int], device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
    """
    Get cached UltraViCo bias for given shape.

    Caches the bias tensor to avoid recomputation.
    """
    global _cached_bias, _cached_shape

    config = _ultravico_config
    if config is None or not config.enabled:
        return None

    # Check if we can use cached bias
    if _cached_bias is not None and _cached_shape == shape:
        if _cached_bias.device == device:
            return _cached_bias.to(dtype)

    # Create new bias
    _cached_bias = create_ultravico_bias_for_shape(shape, config, device, dtype)
    _cached_shape = shape

    return _cached_bias


def get_ultravico_bias_auto(seq_len: int, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
    """
    Get UltraViCo bias using stored visual shape.

    This is called from attention when we don't have explicit shape info.
    Returns None if UltraViCo is disabled or shape doesn't match.
    """
    if not is_ultravico_enabled():
        return None

    shape = _current_visual_shape
    if shape is None:
        return None

    # Verify sequence length matches shape
    expected_len = shape[0] * shape[1] * shape[2]
    if seq_len != expected_len:
        # Shape mismatch - might be text attention or partial sequence
        return None

    return get_ultravico_bias(shape, device, dtype)


def identify_harmonic_period(
    base: float = 10000.0,
    dim: int = 16,
    training_frames: int = 31
) -> Tuple[int, int]:
    """
    Identify the intrinsic frequency and its period for RoPE.

    Based on RIFLEx paper Eq. (4) and (7).

    Args:
        base: RoPE base frequency (theta)
        dim: Dimension of temporal RoPE
        training_frames: Training window in latent frames

    Returns:
        Tuple of (k, period) where k is frequency index and period is in frames
    """
    periods = []
    for j in range(dim // 2):
        theta_j = 1.0 / (base ** (2 * j / dim))
        period_j = int(round(2 * math.pi / theta_j))
        periods.append(period_j)

    # Find frequency with period closest to training_frames
    diffs = [abs(p - training_frames) for p in periods]
    k = diffs.index(min(diffs))

    return k, periods[k]
