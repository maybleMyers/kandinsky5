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

MEMORY OPTIMIZATION:
    For long videos, we use flex_attention with score_mod instead of materializing
    the full [seq_len, seq_len] bias matrix, which would be too large.
"""

import math
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Callable
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


# Global config instance (can be set from CLI)
_ultravico_config: Optional[UltraViCoConfig] = None
_current_visual_shape: Optional[Tuple[int, int, int]] = None


def set_ultravico_config(config: UltraViCoConfig):
    """Set global UltraViCo configuration."""
    global _ultravico_config
    _ultravico_config = config


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


def create_ultravico_score_mod(
    height: int,
    width: int,
    config: UltraViCoConfig
) -> Callable:
    """
    Create a score_mod function for flex_attention that applies UltraViCo decay.

    This is memory-efficient as it computes decay on-the-fly without materializing
    the full [seq_len, seq_len] matrix.

    Args:
        height: Spatial height in latent space (after patching)
        width: Spatial width in latent space (after patching)
        config: UltraViCo configuration

    Returns:
        score_mod function for flex_attention
    """
    hw = height * width
    window_radius = config.training_frames // 2
    log_alpha = math.log(config.alpha) if config.alpha < 1.0 else 0.0
    log_beta = math.log(config.beta) if config.beta < 1.0 else 0.0
    suppress_harmonics = config.suppress_harmonics
    gamma = config.gamma
    period = config.harmonic_period if config.harmonic_period else config.training_frames

    def score_mod(score, batch, head, q_idx, kv_idx):
        # Compute temporal positions from flattened indices
        t_q = q_idx // hw
        t_kv = kv_idx // hw

        # Compute temporal distance
        t_dist = torch.abs(t_q - t_kv)

        # Check if out of training window
        out_of_window = t_dist > window_radius

        # Apply alpha decay to out-of-window
        decay = torch.where(out_of_window, log_alpha, 0.0)

        # Apply beta decay to harmonic positions if enabled
        if suppress_harmonics and log_beta < log_alpha:
            # Check multiple harmonics
            near_harmonic = torch.zeros_like(out_of_window)
            for m in range(1, 10):  # Check up to 10 harmonic periods
                harmonic_center = m * period
                near_this_harmonic = (t_dist >= harmonic_center - gamma) & (t_dist <= harmonic_center + gamma)
                near_harmonic = near_harmonic | near_this_harmonic

            # Apply stronger decay at harmonic positions (only if out of window)
            harmonic_risk = near_harmonic & out_of_window
            decay = torch.where(harmonic_risk, log_beta, decay)

        return score + decay

    return score_mod


def get_ultravico_score_mod() -> Optional[Callable]:
    """
    Get UltraViCo score_mod function for current visual shape.

    Returns None if UltraViCo is disabled or shape not set.
    """
    if not is_ultravico_enabled():
        return None

    shape = _current_visual_shape
    if shape is None:
        return None

    config = _ultravico_config
    _, height, width = shape

    return create_ultravico_score_mod(height, width, config)


def get_ultravico_params() -> Optional[Tuple[int, int, float, float, bool, int, int]]:
    """
    Get UltraViCo parameters for inline computation in attention.

    Returns tuple of (hw, window_radius, log_alpha, log_beta, suppress_harmonics, gamma, period)
    or None if disabled.
    """
    if not is_ultravico_enabled():
        return None

    shape = _current_visual_shape
    if shape is None:
        return None

    config = _ultravico_config
    _, height, width = shape

    hw = height * width
    window_radius = config.training_frames // 2
    log_alpha = math.log(config.alpha) if config.alpha < 1.0 else 0.0
    log_beta = math.log(config.beta) if config.beta < 1.0 else 0.0
    period = config.harmonic_period if config.harmonic_period else config.training_frames

    return (hw, window_radius, log_alpha, log_beta, config.suppress_harmonics, config.gamma, period)


def apply_ultravico_to_scores(
    scores: torch.Tensor,
    seq_len: int,
    params: Tuple[int, int, float, float, bool, int, int]
) -> torch.Tensor:
    """
    Apply UltraViCo decay to attention scores in-place (memory efficient).

    Args:
        scores: Attention scores tensor [..., seq_len, seq_len]
        seq_len: Sequence length
        params: UltraViCo parameters from get_ultravico_params()

    Returns:
        Modified scores tensor
    """
    hw, window_radius, log_alpha, log_beta, suppress_harmonics, gamma, period = params

    # Create position indices
    device = scores.device
    idx = torch.arange(seq_len, device=device)

    # Compute temporal positions
    t_pos = idx // hw  # [seq_len]

    # Compute temporal distance matrix efficiently using broadcasting
    # t_dist[i, j] = |t_i - t_j|
    t_dist = torch.abs(t_pos.unsqueeze(1) - t_pos.unsqueeze(0))  # [seq_len, seq_len]

    # Create decay mask
    out_of_window = t_dist > window_radius
    decay = torch.where(out_of_window, log_alpha, 0.0)

    if suppress_harmonics and log_beta < log_alpha:
        near_harmonic = torch.zeros_like(out_of_window)
        for m in range(1, 10):
            harmonic_center = m * period
            near_this = (t_dist >= harmonic_center - gamma) & (t_dist <= harmonic_center + gamma)
            near_harmonic = near_harmonic | near_this

        harmonic_risk = near_harmonic & out_of_window
        decay = torch.where(harmonic_risk, log_beta, decay)

    # Apply decay to scores (broadcast over batch and head dimensions)
    return scores + decay.to(scores.dtype)


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


# ============================================================================
# SDPA Support - Creates bias matrix for scaled_dot_product_attention
# ============================================================================

_cached_sdpa_bias: Optional[torch.Tensor] = None
_cached_sdpa_shape: Optional[Tuple[int, int, int]] = None


def create_ultravico_bias_for_sdpa(
    shape: Tuple[int, int, int],
    config: UltraViCoConfig,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """
    Create UltraViCo bias matrix for SDPA.

    Uses bfloat16 by default to reduce memory (4GB instead of 8GB for long videos).

    Args:
        shape: Visual shape (T, H, W) after patching
        config: UltraViCo configuration
        device: Device for tensor
        dtype: Data type (bfloat16 recommended for memory efficiency)

    Returns:
        Attention bias tensor of shape [1, 1, seq_len, seq_len] for SDPA
    """
    duration, height, width = shape
    seq_len = duration * height * width
    hw = height * width

    # Compute temporal positions
    idx = torch.arange(seq_len, device=device)
    t_pos = idx // hw

    # Compute temporal distance matrix
    t_dist = torch.abs(t_pos.unsqueeze(1) - t_pos.unsqueeze(0))

    # Training window radius
    window_radius = config.training_frames // 2

    # Create bias (0 = no change, negative = decay)
    log_alpha = math.log(config.alpha) if config.alpha < 1.0 else 0.0
    out_of_window = t_dist > window_radius
    bias = torch.where(out_of_window, log_alpha, 0.0).to(dtype)

    # Apply harmonic suppression if enabled
    if config.suppress_harmonics:
        log_beta = math.log(config.beta) if config.beta < 1.0 else 0.0
        period = config.harmonic_period if config.harmonic_period else config.training_frames
        gamma = config.gamma

        if log_beta < log_alpha:
            near_harmonic = torch.zeros_like(out_of_window)
            max_harmonics = min(10, (t_dist.max().item() // period) + 2)
            for m in range(1, int(max_harmonics)):
                harmonic_center = m * period
                near_this = (t_dist >= harmonic_center - gamma) & (t_dist <= harmonic_center + gamma)
                near_harmonic = near_harmonic | near_this

            harmonic_risk = near_harmonic & out_of_window
            bias = torch.where(harmonic_risk, torch.tensor(log_beta, dtype=dtype, device=device), bias)

    # Add batch and head dimensions for SDPA: [1, 1, seq, seq]
    return bias.unsqueeze(0).unsqueeze(0)


def get_ultravico_sdpa_bias(
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16
) -> Optional[torch.Tensor]:
    """
    Get cached UltraViCo bias for SDPA.

    Returns None if UltraViCo is disabled or shape doesn't match.
    """
    global _cached_sdpa_bias, _cached_sdpa_shape

    if not is_ultravico_enabled():
        return None

    shape = _current_visual_shape
    if shape is None:
        return None

    # Verify sequence length matches
    expected_len = shape[0] * shape[1] * shape[2]
    if seq_len != expected_len:
        return None

    config = _ultravico_config

    # Check cache
    if _cached_sdpa_bias is not None and _cached_sdpa_shape == shape:
        if _cached_sdpa_bias.device == device:
            return _cached_sdpa_bias.to(dtype)

    # Create new bias
    print(f"[UltraViCo] Creating SDPA bias for shape {shape}, seq_len={seq_len}")
    _cached_sdpa_bias = create_ultravico_bias_for_sdpa(shape, config, device, dtype)
    _cached_sdpa_shape = shape

    return _cached_sdpa_bias
