"""
UltraViCo Triton Kernel - Memory-efficient attention with temporal decay.

Computes attention with UltraViCo decay on-the-fly without materializing bias matrix.
"""

import math
import torch
import triton
import triton.language as tl
from typing import Optional, Tuple


@triton.jit
def _ultravico_attn_fwd_kernel(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    # UltraViCo params
    hw: tl.constexpr,  # height * width (spatial size per frame)
    window_radius: tl.constexpr,  # training_frames // 2
    log_alpha: tl.constexpr,  # log(alpha) for out-of-window decay
    # Attention params
    seq_len,
    head_dim: tl.constexpr,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    UltraViCo attention forward kernel.

    Applies temporal decay to attention scores on-the-fly:
    - Tokens within training window: no decay
    - Tokens outside window: multiply by alpha (add log_alpha to logits)
    """
    # Program IDs
    pid_b = tl.program_id(0)  # batch
    pid_h = tl.program_id(1)  # head
    pid_m = tl.program_id(2)  # query block

    # Offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers to Q block
    q_ptrs = Q + pid_b * stride_qb + pid_h * stride_qh + \
             offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk

    # Initialize accumulator and max for online softmax
    acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Load Q block
    q_mask = offs_m[:, None] < seq_len
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Compute temporal position of query tokens
    t_q = offs_m // hw

    # Iterate over K, V blocks
    for start_n in range(0, seq_len, BLOCK_N):
        curr_offs_n = start_n + offs_n

        # Load K block
        k_ptrs = K + pid_b * stride_kb + pid_h * stride_kh + \
                 curr_offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kk
        k_mask = curr_offs_n[None, :] < seq_len
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # Compute attention scores: Q @ K^T
        scores = tl.dot(q, k) * scale  # [BLOCK_M, BLOCK_N]

        # Compute temporal positions of key tokens
        t_kv = curr_offs_n

        # Compute temporal distance and apply UltraViCo decay
        t_kv_broadcast = t_kv[None, :] // hw
        t_dist = tl.abs(t_q[:, None] - t_kv_broadcast)
        out_of_window = t_dist > window_radius

        # Apply decay: add log_alpha where out of window
        decay = tl.where(out_of_window, log_alpha, 0.0)
        scores = scores + decay

        # Mask invalid positions
        valid_mask = (offs_m[:, None] < seq_len) & (curr_offs_n[None, :] < seq_len)
        scores = tl.where(valid_mask, scores, float('-inf'))

        # Online softmax update
        m_ij = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha_scale = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        l_ij = tl.sum(p, axis=1)
        l_new = alpha_scale * l_i + l_ij

        # Load V block
        v_ptrs = V + pid_b * stride_vb + pid_h * stride_vh + \
                 curr_offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=curr_offs_n[:, None] < seq_len, other=0.0)

        # Update accumulator
        acc = acc * alpha_scale[:, None] + tl.dot(p.to(v.dtype), v)

        # Update running max and sum
        m_i = m_new
        l_i = l_new

    # Normalize
    acc = acc / l_i[:, None]

    # Store output
    out_ptrs = Out + pid_b * stride_ob + pid_h * stride_oh + \
               offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    out_mask = offs_m[:, None] < seq_len
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=out_mask)


def ultravico_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    hw: int,
    window_radius: int,
    log_alpha: float,
) -> torch.Tensor:
    """
    UltraViCo attention with temporal decay.

    Args:
        q: Query tensor [batch, heads, seq_len, head_dim]
        k: Key tensor [batch, heads, seq_len, head_dim]
        v: Value tensor [batch, heads, seq_len, head_dim]
        hw: Spatial size (height * width after patching)
        window_radius: Training window radius (training_frames // 2)
        log_alpha: Log of decay factor for out-of-window attention

    Returns:
        Output tensor [batch, heads, seq_len, head_dim]
    """
    batch, heads, seq_len, head_dim = q.shape

    # Ensure contiguous
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    # Output tensor
    out = torch.empty_like(q)

    # Scale factor
    scale = 1.0 / math.sqrt(head_dim)

    # Block sizes
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = head_dim

    # Grid
    grid = (batch, heads, triton.cdiv(seq_len, BLOCK_M))

    # Launch kernel
    _ultravico_attn_fwd_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        hw=hw,
        window_radius=window_radius,
        log_alpha=log_alpha,
        seq_len=seq_len,
        head_dim=BLOCK_K,
        scale=scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return out


class UltraViCoAttention(torch.nn.Module):
    """
    UltraViCo attention module with Triton kernel.

    Drop-in replacement for standard attention that applies temporal decay.
    """

    def __init__(self):
        super().__init__()
        self._params = None

    def set_params(self, hw: int, window_radius: int, log_alpha: float):
        """Set UltraViCo parameters."""
        self._params = (hw, window_radius, log_alpha)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            q, k, v: [batch, heads, seq_len, head_dim] or [batch, seq_len, heads, head_dim]

        Returns:
            Output tensor same shape as input
        """
        if self._params is None:
            # Fall back to standard attention if params not set
            return torch.nn.functional.scaled_dot_product_attention(q, k, v)

        hw, window_radius, log_alpha = self._params

        # Handle different input formats
        needs_transpose = q.shape[2] != q.shape[1]  # [B, S, H, D] vs [B, H, S, D]
        if needs_transpose:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

        out = ultravico_attention(q, k, v, hw, window_radius, log_alpha)

        if needs_transpose:
            out = out.transpose(1, 2)

        return out


# Global instance for easy access
_ultravico_attn: Optional[UltraViCoAttention] = None


def get_ultravico_triton_attention() -> UltraViCoAttention:
    """Get or create global UltraViCo attention instance."""
    global _ultravico_attn
    if _ultravico_attn is None:
        _ultravico_attn = UltraViCoAttention()
    return _ultravico_attn


def setup_ultravico_triton(hw: int, window_radius: int, alpha: float):
    """Setup UltraViCo Triton attention with given parameters."""
    import math
    attn = get_ultravico_triton_attention()
    log_alpha = math.log(alpha) if alpha < 1.0 else 0.0
    attn.set_params(hw, window_radius, log_alpha)
    return attn
