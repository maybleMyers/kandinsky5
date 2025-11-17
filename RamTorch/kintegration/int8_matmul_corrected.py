import torch
import triton
import triton.language as tl
from triton import Config
from typing import Tuple


"""
simplified explanation of the scaled int8 matmul algorithm
adopted from deepseek scaled FP8 matmul and jetfire paper
https://arxiv.org/abs/2403.12422
https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/kernel.py

                                                     N dimension →  
                                               INT8 weights                 scaler per block
                                               ┌-----┬-----┬─────┬─────┐    ┌-----┬-----┬─────┬─────┐
                                               : b00 : b01 : b02 | b03 |    :     :     :     |     |
                                               ├-----┼-----┼─────┼─────┤    :b_s00:b_s10:b_s20|b_s30|
                                           K   : b10 : b11 : b12 | b13 |    :     :     :     |     |
                                          dim  ├-----┼-----┼─────┼─────┤    ├-----┼-----┼─────┼─────┤
                                           ↓   | b20 | b21 | b22 | b23 |    |     |     |     |     |
                                               ├─────┼─────┼─────┼─────┤    |b_s01|b_s11|b_s21|b_s31|
                                               | b30 | b31 | b32 | b33 |    |     |     |     |     |
                                               └─────┴─────┴─────┴─────┘    └─────┴─────┴─────┴─────┘
                                               ┌-----┬-----┐
                                               : b00 : b01 :
     ├─── blk ───┤                             ├-----┼-----┤
                                               : b10 : b11 :
            K dimension →                      └-----┴-----┘                                
     INT8 activations
     ┌-----┬-----┬─────┬─────┐   ┌-----┬-----┐ ┌-----┬-----┐   ┌-----------┐   ┌-----┬-----┐   ┌-----┬-----┐
     : a00 : a01 : a02 | a03 |   : a00 : a01 : :  @  :  @  :   :   a_s00   :   :     :     :   :acc00:acc01:
     ├-----┼-----┼─────┼─────┤   ├-----┼-----┤ ├-----┼-----┤ * ├-----------┤ * :b_s00:b_s10: = ├-----┼-----┤ 
 M   : a10 : a11 : a12 | a13 |   : a10 : a11 : :  @  :  @  :   :   a_s10   :   :     :     :   :acc10:acc11:
dim  ├-----┼-----┼─────┼─────┤   └-----┴-----┘ └-----┴-----┘   └-----------┘   └-----┴-----┘   └-----┴-----┘
 ↓   | a20 | a21 | a22 | a23 |   INT8 matmul acc in INT32      rescale the FP32 intermediate   accumulate
     ├─────┼─────┼─────┼─────┤   then cast to FP32             "rank 1" hadamard scaler        intermediate
     | a30 | a31 | a32 | a33 |  
     └─────┴─────┴─────┴─────┘  
     scaler per block
     ┌-----------┬───────────┐
     :   a_s00   :   a_s01   |
     ├-----------┼───────────┤
     :   a_s10   :   a_s11   |
     ├-----------┼───────────┤
     |   a_s20   |   a_s21   |
     ├───────────┼───────────┤
     |   a_s30   |   a_s31   |
     └───────────┴───────────┘
"""


@triton.jit
def act_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr):
    """
    Quantizes the input tensor `x_ptr` and stores the result in `y_ptr` and the scaling factor in `s_ptr`.

    Args:
        x_ptr (triton.Pointer): Pointer to the input tensor.
        y_ptr (triton.Pointer): Pointer to the output tensor where quantized values will be stored.
        s_ptr (triton.Pointer): Pointer to the output tensor where scaling factors will be stored.
        BLOCK_SIZE (tl.constexpr): The size of the block to be processed by each program instance.

    Returns:
        None
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs).to(tl.float32)
    amax = tl.max(tl.abs(x))  # reduction
    # amax = tl.maximum(amax, 1e-4) # clamp to 1e-4
    s = amax / 127.0
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y)
    tl.store(s_ptr + pid, s)


def act_quant(
    x: torch.Tensor, block_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using block-wise quantization.

    Args:
        x (torch.Tensor): The input tensor to be quantized. Must be contiguous and its last dimension size must be divisible by `block_size`.
        block_size (int, optional): The size of the blocks to be used for quantization. Default is 128.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The quantized tensor with dtype `torch.int8`.
            - A tensor of scaling factors with dtype `torch.float32`.
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert (
        x.size(-1) % block_size == 0
    ), f"Last dimension size must be divisible by block_size (block_size={block_size})"
    y = torch.empty_like(x, dtype=torch.int8)
    s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
    act_quant_kernel[grid](x, y, s, BLOCK_SIZE=block_size)
    return y, s


@triton.jit
def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    """
    Dequantizes weights using the provided scaling factors and stores the result.

    Args:
        x_ptr (tl.pointer): Pointer to the quantized weights.
        s_ptr (tl.pointer): Pointer to the scaling factors.
        y_ptr (tl.pointer): Pointer to the output buffer for dequantized weights.
        M (int): Number of rows in the weight matrix.
        N (int): Number of columns in the weight matrix.
        BLOCK_SIZE (tl.constexpr): Size of the block for tiling.

    Returns:
        None
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + pid_m * n + pid_n)
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)


def weight_dequant(
    x: torch.Tensor, s: torch.Tensor, block_size: int = 128
) -> torch.Tensor:
    """
    Dequantizes the given weight tensor using the provided scale tensor.

    Args:
        x (torch.Tensor): The quantized weight tensor of shape (M, N).
        s (torch.Tensor): The scale tensor of shape (M//block_size, N//block_size).
        block_size (int, optional): The block size to use for dequantization. Defaults to 128.

    Returns:
        torch.Tensor: The dequantized weight tensor of the same shape as `x`.

    Raises:
        AssertionError: If `x` or `s` are not contiguous or if their dimensions are not 2.
    """
    assert x.is_contiguous() and s.is_contiguous(), "Input tensors must be contiguous"
    assert x.dim() == 2 and s.dim() == 2, "Input tensors must have 2 dimensions"
    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.get_default_dtype())
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE"]),
        triton.cdiv(N, meta["BLOCK_SIZE"]),
    )
    weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
    return y


# matmul intermediate block size is hardcoded to 128
int8_gemm_configs = [
    Config(
        {"BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n, "BLOCK_SIZE_K": 128},
        num_stages=num_stages,
        num_warps=8,
    )
    for block_m in [16, 32, 64]
    for block_n in [32, 64, 128]
    for num_stages in [3, 4, 5, 6]
]


@triton.autotune(configs=int8_gemm_configs, key=["N", "K"])
@triton.jit
def int8_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_s_ptr,
    b_s_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Performs a matrix multiplication operation on INT8 matrices with scaling factors.

    Args:
        a_ptr (tl.tensor): Pointer to the first input matrix A.
        b_ptr (tl.tensor): Pointer to the second input matrix B.
        c_ptr (tl.tensor): Pointer to the output matrix C.
        a_s_ptr (tl.tensor): Pointer to the scaling factors for matrix A.
        b_s_ptr (tl.tensor): Pointer to the scaling factors for matrix B.
        M (int): Number of rows in matrix A and C.
        N (tl.constexpr): Number of columns in matrix B and C.
        K (tl.constexpr): Number of columns in matrix A and rows in matrix B.
        BLOCK_SIZE_M (tl.constexpr): Block size for the M dimension.
        BLOCK_SIZE_N (tl.constexpr): Block size for the N dimension.
        BLOCK_SIZE_K (tl.constexpr): Block size for the K dimension (must be 128).

    Returns:
        None
    
    Note:
        The recent update corrected the b_s_ptrs calculation to use offs_n * k instead of
        (offs_n // BLOCK_SIZE_K) * k for proper scaling factor indexing.
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_SIZE_K)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize pointers
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]
    a_s_ptrs = a_s_ptr + offs_m * k
    
    # CORRECTED: Use offs_n * k instead of (offs_n // BLOCK_SIZE_K) * k
    b_s_ptrs = b_s_ptr + offs_n * k
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for i in range(k):
        # Load INT8 values
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0.0)
        
        # Load scaling factors
        a_s = tl.load(a_s_ptrs)
        b_s = tl.load(b_s_ptrs)
        
        # Compute INT8 dot product and scale
        dot_prod = tl.dot(a, b)
        accumulator += dot_prod.to(tl.float32) * a_s[:, None] * b_s[None, :]
        
        # Advance pointers
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
        a_s_ptrs += 1
        b_s_ptrs += 1
    
    # Store result
    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


def int8_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor):
    """
    Perform a matrix multiplication using INT8 precision with Triton acceleration.

    Args:
        a (torch.Tensor): The first input matrix (activations), must be contiguous.
        a_s (torch.Tensor): The scaling factors for the first input matrix, must be contiguous.
        b (torch.Tensor): The second input matrix (weights), must be contiguous.
        b_s (torch.Tensor): The scaling factors for the second input matrix, must be contiguous.

    Returns:
        torch.Tensor: The result of the matrix multiplication in FP32.
    
    Example:
        >>> # Quantize activations
        >>> x_int8, x_scales = act_quant(x, block_size=128)
        >>> # Assume weights are pre-quantized
        >>> output = int8_gemm(x_int8, x_scales, weight_int8, weight_scales)
    """
    assert a.is_contiguous() and b.is_contiguous(), "Input tensors must be contiguous"
    assert (
        a_s.is_contiguous() and b_s.is_contiguous()
    ), "Scaling factor tensors must be contiguous"
    
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.get_default_dtype())
    
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    
    int8_gemm_kernel[grid](a, b, c, a_s, b_s, M, N, K)
    
    return c


# Additional helper class for easier integration
class Int8LinearFunction(torch.autograd.Function):
    """
    Custom autograd function for INT8 linear layers.
    Enables gradient computation through quantized layers.
    """
    
    @staticmethod
    def forward(ctx, input, weight_int8, weight_scales, bias=None, block_size=128):
        """Forward pass using INT8 computation"""
        # Quantize input
        input_int8, input_scales = act_quant(input, block_size=block_size)
        
        # Perform INT8 matmul
        output = int8_gemm(input_int8, input_scales, weight_int8, weight_scales)
        
        # Add bias if present
        if bias is not None:
            output = output + bias
        
        # Save for backward (if needed)
        ctx.save_for_backward(input, weight_int8, weight_scales)
        ctx.block_size = block_size
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass through INT8 layer"""
        input, weight_int8, weight_scales = ctx.saved_tensors
        block_size = ctx.block_size
        
        # Dequantize weight for gradient computation
        weight = weight_dequant(weight_int8, weight_scales, block_size=block_size)
        
        # Compute gradients
        grad_input = grad_output @ weight.t()
        grad_weight = grad_output.t() @ input
        
        grad_bias = grad_output.sum(0) if ctx.needs_input_grad[3] else None
        
        return grad_input, None, None, grad_bias, None


def int8_linear(input, weight_int8, weight_scales, bias=None, block_size=128):
    """
    Functional interface for INT8 linear layer.
    
    Args:
        input: Input tensor
        weight_int8: Quantized weight in INT8
        weight_scales: Scaling factors for weight
        bias: Optional bias tensor
        block_size: Block size for quantization (must be 128)
    
    Returns:
        Output tensor in FP32
    """
    return Int8LinearFunction.apply(input, weight_int8, weight_scales, bias, block_size)
