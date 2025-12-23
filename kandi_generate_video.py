import argparse
import time
import warnings
import logging
import os
import tempfile
import sys

import numpy as np
import torch
from PIL import Image

# Early parse --no_compile to set the flag before importing kandinsky
def _early_parse_no_compile():
    for i, arg in enumerate(sys.argv):
        if arg == '--no_compile':
            return True
    return False

# Early parse SDNQ flags to enable optimal int8 performance BEFORE importing SDNQ
def _early_parse_sdnq_flags():
    """Parse SDNQ-related flags early to set env vars before module import.

    CRITICAL: SDNQ module evaluates these environment variables at import time.
    Setting them after import has no effect.
    """
    use_sdnq = '--use_sdnq' in sys.argv
    no_triton_mm = '--no_sdnq_triton_mm' in sys.argv
    no_compile = '--no_sdnq_compile' in sys.argv

    if use_sdnq:
        # Handle Triton int8 matmul kernel setting
        if no_triton_mm:
            os.environ["SDNQ_USE_TRITON_MM"] = "0"
            print("SDNQ: Using torch._int_mm (Triton MM disabled)")
        elif os.environ.get("SDNQ_USE_TRITON_MM") is None:
            # Enable Triton int8 matmul kernel for optimal tensor core utilization on CUDA
            # Default SDNQ only enables Triton MM for RDNA2/ZLUDA, but it's faster on 4090/5090 too
            os.environ["SDNQ_USE_TRITON_MM"] = "1"
            print("SDNQ: Enabling Triton int8 matmul kernel for optimal CUDA performance")

        # Handle torch.compile setting
        if no_compile:
            os.environ["SDNQ_USE_TORCH_COMPILE"] = "0"
            print("SDNQ: torch.compile disabled for faster startup")
        elif os.environ.get("SDNQ_USE_TORCH_COMPILE") is None:
            # Enable torch.compile for SDNQ dequantization (if Triton is available)
            os.environ["SDNQ_USE_TORCH_COMPILE"] = "1"
            print("SDNQ: Enabling torch.compile for dequantization kernels")

# Must be called before any SDNQ-related imports
_early_parse_sdnq_flags()

# Set global compile flag before importing kandinsky modules
import kandinsky.models.compile_config as compile_config
_no_compile = _early_parse_no_compile()
compile_config.USE_TORCH_COMPILE = not _no_compile
if _no_compile:
    print("torch.compile() disabled for faster startup")

from kandinsky import get_T2V_pipeline, get_I2V_pipeline, get_I2V_pipeline_with_block_swap, get_T2V_pipeline_with_block_swap, get_T2I_pipeline
from kandinsky.generation_utils import generate_sample_from_checkpoint, generate_sample_i2v_from_checkpoint, generate_sample_v2v, generate_sample_v2v_join
from kandinsky.i2v_pipeline import (
    get_conditioning_frames_from_video,
    get_conditioning_frames_from_two_videos,
    get_conditioning_latents_from_two_images,
    get_conditioning_video_and_image,
    encode_video_to_latents,
    Kandinsky5DenoisePipeline
)
from kandinsky.generation_utils import generate_sample_denoise
try:
    from scripts.latentpreviewer import LatentPreviewer
except ImportError:
    LatentPreviewer = None



def disable_warnings():
    warnings.filterwarnings("ignore")
    logging.getLogger("torch").setLevel(logging.ERROR)
    torch._logging.set_logs(
        dynamo=logging.ERROR,
        dynamic=logging.ERROR,
        aot=logging.ERROR,
        inductor=logging.ERROR,
        guards=False,
        recompiles=False
    )


def resize_image_to_resolution(image_path, target_width, target_height, alignment=32):
    """
    Resize image to target resolution while maintaining aspect ratio and ensuring
    dimensions are multiples of alignment.

    Args:
        image_path: Path to the input image
        target_width: Target width (will be rounded to alignment)
        target_height: Target height (will be rounded to alignment)
        alignment: Pixel alignment (32 for standard, 128 for NABLA)

    Returns:
        Path to the resized image (temporary file)

    Note:
        NABLA attention requires 128-pixel alignment due to fractal flattening.
        Standard attention only requires 32-pixel alignment.
    """
    try:
        img = Image.open(image_path)
        original_width, original_height = img.size

        # Ensure target dimensions are multiples of alignment
        target_width = (target_width // alignment) * alignment
        target_height = (target_height // alignment) * alignment
        target_width = max(alignment * 2, target_width)  # Minimum 2x alignment
        target_height = max(alignment * 2, target_height)

        # Check if resizing is needed
        if original_width == target_width and original_height == target_height:
            print(f"Image already at target resolution: {target_width}x{target_height}")
            return image_path

        print(f"Resizing image from {original_width}x{original_height} to {target_width}x{target_height}")

        # Resize the image
        resized_img = img.resize((target_width, target_height), Image.LANCZOS)

        # Save to temporary file
        temp_dir = tempfile.gettempdir()
        temp_filename = f"resized_input_{os.path.basename(image_path)}"
        temp_path = os.path.join(temp_dir, temp_filename)

        # Preserve the image format
        resized_img.save(temp_path, format=img.format if img.format else 'PNG')
        print(f"Resized image saved to: {temp_path}")

        return temp_path

    except Exception as e:
        print(f"Error resizing image: {e}")
        print(f"Using original image: {image_path}")
        return image_path


def normalize_join_frames(video1, video2, num_frames):
    """
    Normalize frames at the join point between two videos to reduce flash/discontinuity.

    This function:
    1. Computes color statistics (mean, std) for boundary regions
    2. Applies color matching to align the second video's colors with the first
    3. Applies cross-fade blending in the overlap region

    Args:
        video1: First video tensor [num_frames, H, W, C] (float, 0-255 range)
        video2: Second video tensor [num_frames, H, W, C] (float, 0-255 range)
        num_frames: Number of frames to blend at the boundary

    Returns:
        video1: Unchanged first video
        video2: Color-matched and blended second video
    """
    if num_frames <= 0:
        return video1, video2

    # Ensure we have enough frames
    num_frames = min(num_frames, video1.shape[0], video2.shape[0])

    # Get boundary regions for statistics
    v1_end = video1[-num_frames:]  # Last N frames of video1
    v2_start = video2[:num_frames]  # First N frames of video2

    # Compute per-channel statistics for color matching
    # Using the boundary frames to compute mean and std
    v1_mean = v1_end.mean(dim=(0, 1, 2), keepdim=True)  # [1, 1, 1, C]
    v1_std = v1_end.std(dim=(0, 1, 2), keepdim=True) + 1e-6
    v2_mean = v2_start.mean(dim=(0, 1, 2), keepdim=True)
    v2_std = v2_start.std(dim=(0, 1, 2), keepdim=True) + 1e-6

    # Apply color matching to entire video2
    # Transform: (x - mean2) / std2 * std1 + mean1
    video2_matched = (video2 - v2_mean) / v2_std * v1_std + v1_mean
    video2_matched = video2_matched.clamp(0, 255)

    # Apply cross-fade blending in the overlap region
    # Modify the first num_frames of video2 to blend with end of video1
    for i in range(num_frames):
        # Alpha goes from 1 (favor video1) to 0 (favor video2)
        alpha = 1.0 - (i + 1) / (num_frames + 1)
        video2_matched[i] = alpha * video1[-(num_frames - i)] + (1 - alpha) * video2_matched[i]

    print(f">>> Frame normalization: Applied color matching and {num_frames}-frame cross-fade")

    return video1, video2_matched


def normalize_join_frames_triple(video1, middle, video2, num_frames):
    """
    Normalize frames at both join points in a three-video concatenation.

    Used for video join mode: video1 + middle + video2

    Args:
        video1: First video tensor [num_frames, H, W, C]
        middle: Middle generated video tensor [num_frames, H, W, C]
        video2: Second video tensor [num_frames, H, W, C]
        num_frames: Number of frames to blend at each boundary

    Returns:
        video1, middle, video2: Normalized tensors with smooth transitions
    """
    if num_frames <= 0:
        return video1, middle, video2

    # First junction: video1 -> middle
    video1, middle = normalize_join_frames(video1, middle, num_frames)

    # Second junction: middle -> video2
    # For this, we need to normalize video2 to match middle's ending
    middle, video2 = normalize_join_frames(middle, video2, num_frames)

    return video1, middle, video2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a video using Kandinsky 5"
    )
    parser.add_argument(
        '--local-rank',
        type=int,
        help='local rank'
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/config_5s_sft.yaml",
        help="The config file of the model"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The dragon soars into the sunset sky.",
        help="The prompt to generate video"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="./assets/test_image.jpg",
        help="The input image for image-to-video generation"
    )
    parser.add_argument(
        "--end_image",
        type=str,
        default=None,
        help="Ending image for image-to-video generation. When provided with --image, generates a video transitioning from the start image to the end image."
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Input video for video continuation (overrides --image)"
    )
    parser.add_argument(
        "--video2",
        type=str,
        default=None,
        help="Second input video for video joining mode. When provided with --video, creates a transition between the two videos."
    )
    parser.add_argument(
        "--num_cond_frames",
        type=int,
        default=4,
        help="Number of last frames to use as conditioning for video continuation (or frames from each video in join mode)"
    )
    parser.add_argument(
        "--normalize_frames",
        type=int,
        default=0,
        help="Number of frames to blend at join points (0=disabled). Smoothly transitions color/brightness at video boundaries."
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="Static, 2D cartoon, cartoon, 2d animation, paintings, images, worst quality, low quality, ugly, deformed, walking backwards",
        help="Negative prompt for classifier-free guidance"
    )
    parser.add_argument(
        "--clip_prompt",
        type=str,
        default=None,
        help="Separate prompt for CLIP encoder (if not provided, uses main prompt)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=768,
        help="Width of the video in pixels"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Height of the video in pixels"
    )
    parser.add_argument(
        "--video_duration",
        type=int,
        default=5,
        help="Duratioin of the video in seconds"
    )
    parser.add_argument(
        "--expand_prompt",
        type=int,
        default=1,
        help="Whether to use prompt expansion."
    )
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=None,
        help="The sampling steps number."
    )
    parser.add_argument(
        "--guidance_weight",
        type=float,
        default=None,
        help="Guidance weight."
    )
    parser.add_argument(
        "--scheduler_scale",
        type=float,
        default=5.0,
        help="Scheduler scale."
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="./test.mp4",
        help="Name of the resulting file"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1137,
        help="Seed for the random number generator"
    )

    parser.add_argument(
        "--offload",
        action='store_true',
        default=False,
        help="Offload models to save memory or not"
    )
    parser.add_argument(
        "--magcache",
        action='store_true',
        default=False,
        help="Using MagCache (for 50 steps models only)"
    )
    parser.add_argument(
        "--qwen_quantization",
        action='store_true',
        default=False,
        help="Use quantized Qwen2.5-VL model (4-bit quantization)"
    )
    parser.add_argument(
        "--attention_engine",
        type=str,
        default="auto",
        help="Name of the full attention algorithm to use for <=5 second generation",
        choices=["flash_attention_2", "flash_attention_3", "sdpa", "sage", "auto"]
    )
    parser.add_argument(
        "--enable_block_swap",
        action='store_true',
        default=False,
        help="Enable block swapping for large models (e.g., 20B) to fit in limited VRAM"
    )
    parser.add_argument(
        "--blocks_in_memory",
        type=int,
        default=6,
        help="Number of transformer blocks to keep in GPU memory when using block swapping"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16", "fp8_scaled"],
        help="Data type for model weights (default: bfloat16). Use bfloat16 for best memory efficiency with minimal quality loss. Use fp8_scaled for maximum memory savings (~50%% vs bf16). This sets all dtypes if specific ones are not provided."
    )
    parser.add_argument(
        "--text_encoder_dtype",
        type=str,
        default=None,
        choices=["float32", "float16", "bfloat16", "fp8_scaled"],
        help="Data type specifically for text encoder. If not set, uses --dtype value."
    )
    parser.add_argument(
        "--vae_dtype",
        type=str,
        default=None,
        choices=["float32", "float16", "bfloat16", "fp8_scaled"],
        help="Data type specifically for VAE. If not set, uses --dtype value."
    )
    parser.add_argument(
        "--computation_dtype",
        type=str,
        default=None,
        choices=["float32", "float16", "bfloat16", "fp8_scaled"],
        help="Data type for activations/computations. If not set, uses --dtype value."
    )
    parser.add_argument(
        "--use_mixed_weights",
        action='store_true',
        default=False,
        help="Use mixed precision weights - preserve fp32 for critical layers (norms, embeddings) while using specified dtype for activations. Prevents dtype conversion errors."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Override DiT model checkpoint path from config. Provide path to your .safetensors file."
    )

    # INT8 quantization configuration (legacy)
    parser.add_argument(
        "--use_int8",
        action='store_true',
        default=False,
        help="Use legacy INT8 quantization for linear layers. Consider --use_sdnq for better performance."
    )
    parser.add_argument(
        "--int8_block_size",
        type=int,
        default=128,
        help="Block size for legacy INT8 quantization (must be 128 for Triton kernels, default: 128)"
    )

    # SDNQ quantization configuration (recommended)
    parser.add_argument(
        "--use_sdnq",
        action='store_true',
        default=False,
        help="Use SDNQ quantization with auto-tuned Triton kernels (20-40%% faster than legacy INT8)"
    )
    parser.add_argument(
        "--sdnq_weights_dtype",
        type=str,
        default="int8",
        choices=["int8", "fp8", "int4"],
        help="SDNQ weight storage dtype (default: int8). int8=best balance, fp8=H100+, int4=experimental"
    )
    parser.add_argument(
        "--sdnq_use_quantized_matmul",
        action='store_true',
        default=True,
        help="Use accelerated quantized matmul (default: True). Disable for debugging."
    )
    parser.add_argument(
        "--no_sdnq_quantized_matmul",
        action='store_true',
        default=False,
        help="Disable SDNQ quantized matmul (forces dequantize+fp matmul path)"
    )
    parser.add_argument(
        "--sdnq_triton_mm",
        action='store_true',
        default=True,
        help="Use Triton int8 matmul kernel (default: True for CUDA). Faster on 4090/5090."
    )
    parser.add_argument(
        "--no_sdnq_triton_mm",
        action='store_true',
        default=False,
        help="Disable Triton int8 matmul kernel (fallback to torch._int_mm)"
    )
    parser.add_argument(
        "--sdnq_compile",
        action='store_true',
        default=True,
        help="Enable torch.compile for SDNQ kernels (default: True). Improves performance after warmup."
    )
    parser.add_argument(
        "--no_sdnq_compile",
        action='store_true',
        default=False,
        help="Disable torch.compile for SDNQ kernels (faster startup, slower inference)"
    )

    # NABLA sparse attention configuration
    parser.add_argument(
        "--attention_type",
        type=str,
        default=None,
        choices=["auto", "flash", "nabla"],
        help="Attention type: 'flash' for full attention, 'nabla' for sparse attention, 'auto' uses config default."
    )
    parser.add_argument(
        "--nabla_P",
        type=float,
        default=0.9,
        help="NABLA attention: Top-k probability threshold (default: 0.9)"
    )
    parser.add_argument(
        "--nabla_wT",
        type=int,
        default=11,
        help="NABLA attention: Temporal window size (default: 11 for 10s, 7 for 5s)"
    )
    parser.add_argument(
        "--nabla_wW",
        type=int,
        default=3,
        help="NABLA attention: Width window size (default: 3)"
    )
    parser.add_argument(
        "--nabla_wH",
        type=int,
        default=3,
        help="NABLA attention: Height window size (default: 3)"
    )
    parser.add_argument(
        "--nabla_method",
        type=str,
        default="topcdf",
        choices=["topcdf"],
        help="NABLA attention: Selection method (default: topcdf)"
    )
    parser.add_argument(
        "--nabla_add_sta",
        action='store_true',
        default=True,
        help="NABLA attention: Add spatial-temporal attention (default: True)"
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=None,
        metavar="N",
        help="Enable latent preview every N steps. Generates previews in 'previews' subdirectory."
    )
    parser.add_argument(
        "--preview_suffix",
        type=str,
        default=None,
        help="Unique suffix for preview files to avoid conflicts in concurrent runs."
    )

    # APG (Adaptive Projected Guidance) for video continuation
    parser.add_argument(
        "--use_apg",
        action='store_true',
        default=False,
        help="Enable Adaptive Projected Guidance to reduce color drift in video continuation"
    )
    parser.add_argument(
        "--apg_momentum",
        type=float,
        default=-0.75,
        help="Momentum for APG running average (default: -0.75)"
    )
    parser.add_argument(
        "--apg_norm_threshold",
        type=float,
        default=55.0,
        help="Norm threshold for APG guidance clipping (default: 55.0)"
    )

    # End frame blending for video join modes
    parser.add_argument(
        "--end_blend_weight",
        type=float,
        default=0.0,
        help="Final blend weight for end frames in v2v join mode. 0.0 = use denoised result (smooth transition), 1.0 = use target latent (may cause jump). Default: 0.0"
    )

    # VAE temporal chunking configuration
    parser.add_argument(
        "--vae_temporal_tile_frames",
        type=int,
        default=None,
        help="Temporal chunk size for VAE decode in pixel-space frames (default: 16). Lower values reduce memory usage. Recommended: 12 for moderate memory reduction, 8 for aggressive reduction. Must be divisible by 4."
    )
    parser.add_argument(
        "--vae_temporal_stride_frames",
        type=int,
        default=None,
        help="Temporal stride for VAE decode in pixel-space frames (default: tile_frames - 4). Controls overlap between chunks for smooth blending. If not specified, auto-calculated as tile_frames - 4."
    )
    parser.add_argument(
        "--vae_spatial_tile_height",
        type=int,
        default=None,
        help="Spatial tile height for VAE decode (default: 256). Lower values reduce memory usage but increase processing time."
    )
    parser.add_argument(
        "--vae_spatial_tile_width",
        type=int,
        default=None,
        help="Spatial tile width for VAE decode (default: 256). Lower values reduce memory usage but increase processing time."
    )
    parser.add_argument(
        "--no_compile",
        action='store_true',
        default=False,
        help="Disable torch.compile() for faster startup (2-5 minutes faster) at the cost of slower inference"
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint file to resume generation from"
    )
    parser.add_argument(
        "--save_latents",
        type=str,
        default=None,
        help="Path to save latents before VAE decoding (e.g., latents.pt). Saves all info needed for later decoding."
    )
    parser.add_argument(
        "--decode_from_file",
        type=str,
        default=None,
        help="Path to load and decode previously saved latents. Skips generation and only runs VAE decoding."
    )

    # Video denoise mode
    parser.add_argument(
        "--denoise",
        action='store_true',
        default=False,
        help="Enable video denoise mode. Applies light denoising to smooth video artifacts."
    )
    parser.add_argument(
        "--denoise_strength",
        type=float,
        default=0.2,
        help="Denoise strength (0.1-0.5 typical). Higher = more change. Default: 0.2"
    )

    # LoRA support
    parser.add_argument(
        "--lora_path",
        type=str,
        nargs="*",
        default=None,
        help="Path(s) to LoRA directories containing config_lora.json and lora.safetensors. Multiple LoRAs can be loaded (e.g., --lora_path ./lora1 ./lora2)"
    )
    parser.add_argument(
        "--lora_weight",
        type=float,
        nargs="*",
        default=None,
        help="Weight(s) for each LoRA (0.0-1.0). Must match number of --lora_path entries. Default: 1.0 for each"
    )
    parser.add_argument(
        "--lora_trigger",
        type=str,
        nargs="*",
        default=None,
        help="Override trigger word(s) for each LoRA. If not specified, auto-detected from LoRA metadata"
    )

    # UltraViCo: Attention decay for long video extrapolation
    parser.add_argument(
        "--ultravico",
        action='store_true',
        default=False,
        help="Enable UltraViCo attention decay for long video generation. Helps prevent quality degradation and content repetition when generating videos longer than training length."
    )
    parser.add_argument(
        "--ultravico_alpha",
        type=float,
        default=0.9,
        help="UltraViCo: Decay factor for out-of-window attention (0.85-0.95 recommended). Lower = stronger decay. Default: 0.9"
    )
    parser.add_argument(
        "--ultravico_training_frames",
        type=int,
        default=None,
        help="UltraViCo: Training window in latent frames. Auto-detected from config if not set (5s=31, 10s=61)."
    )
    parser.add_argument(
        "--ultravico_suppress_harmonics",
        action='store_true',
        default=False,
        help="UltraViCo: Enable stronger suppression at harmonic positions. Use if you see content repetition/looping."
    )
    parser.add_argument(
        "--ultravico_beta",
        type=float,
        default=0.6,
        help="UltraViCo: Decay factor for harmonic risk positions (only with --ultravico_suppress_harmonics). Default: 0.6"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    disable_warnings()
    args = parse_args()

    # Convert string dtype to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "fp8_scaled": torch.bfloat16,  # FP8 uses bfloat16 as compute dtype
    }

    # Track which components should use FP8
    use_fp8 = args.dtype == "fp8_scaled"
    use_fp8_text_encoder = args.text_encoder_dtype == "fp8_scaled" if args.text_encoder_dtype else use_fp8
    use_fp8_vae = args.vae_dtype == "fp8_scaled" if args.vae_dtype else use_fp8
    use_fp8_computation = args.computation_dtype == "fp8_scaled" if args.computation_dtype else use_fp8

    # SDNQ quantization settings
    use_sdnq = args.use_sdnq
    sdnq_weights_dtype = args.sdnq_weights_dtype
    sdnq_use_quantized_matmul = args.sdnq_use_quantized_matmul and not args.no_sdnq_quantized_matmul

    # If SDNQ is enabled, disable legacy INT8 and FP8 (they are mutually exclusive)
    if use_sdnq:
        if args.use_int8:
            print("Note: --use_sdnq takes priority over --use_int8. Legacy INT8 disabled.")
        if use_fp8_computation:
            print("Note: --use_sdnq takes priority over fp8_scaled. Legacy FP8 disabled.")
        use_fp8_computation = False

    model_dtype = dtype_map[args.dtype]

    # Set individual component dtypes (fall back to model_dtype if not specified)
    text_encoder_dtype = dtype_map[args.text_encoder_dtype] if args.text_encoder_dtype else model_dtype
    vae_dtype = dtype_map[args.vae_dtype] if args.vae_dtype else model_dtype
    computation_dtype = dtype_map[args.computation_dtype] if args.computation_dtype else model_dtype

    # Build attention config override if attention_type is specified
    attention_config = None
    if args.attention_type and args.attention_type != "auto":
        attention_config = {
            "type": args.attention_type,
            "causal": False,
            "local": False,
            "glob": False,
            "window": 3,
        }
        if args.attention_type == "nabla":
            attention_config.update({
                "P": args.nabla_P,
                "wT": args.nabla_wT,
                "wW": args.nabla_wW,
                "wH": args.nabla_wH,
                "add_sta": args.nabla_add_sta,
                "method": args.nabla_method,
            })

    # Initialize UltraViCo if enabled (for long video extrapolation)
    if args.ultravico:
        from kandinsky.models.ultravico import UltraViCoConfig, set_ultravico_config

        # Auto-detect training frames from config name if not specified
        training_frames = args.ultravico_training_frames
        if training_frames is None:
            if "10s" in args.config:
                training_frames = 61  # 10s = 61 latent frames
            else:
                training_frames = 31  # 5s = 31 latent frames (default)

        ultravico_config = UltraViCoConfig(
            enabled=True,
            training_frames=training_frames,
            alpha=args.ultravico_alpha,
            beta=args.ultravico_beta,
            suppress_harmonics=args.ultravico_suppress_harmonics,
            gamma=4,
        )
        set_ultravico_config(ultravico_config)
        print(f"UltraViCo enabled: training_frames={training_frames}, alpha={args.ultravico_alpha}, "
              f"suppress_harmonics={args.ultravico_suppress_harmonics}")

    # Determine model type from config filename
    is_t2i = "t2i" in args.config.lower()
    is_i2v = "i2v" in args.config.lower()
    is_t2v_pro = "t2v" in args.config.lower() and ("pro" in args.config.lower() or "20b" in args.config.lower())

    if is_t2i:
        # Use T2I pipeline for text-to-image generation
        pipe = get_T2I_pipeline(
            device_map={"dit": "cuda:0", "vae": "cuda:0",
                        "text_embedder": "cuda:0"},
            conf_path=args.config,
            offload=args.offload,
            magcache=args.magcache,
            quantized_qwen=args.qwen_quantization,
            attention_engine=args.attention_engine,
        )
    elif is_i2v:
        if args.enable_block_swap:
            # Use block swapping pipeline for large I2V models
            pipe = get_I2V_pipeline_with_block_swap(
                device_map={"dit": "cuda:0", "vae": "cuda:0",
                            "text_embedder": "cuda:0"},
                conf_path=args.config,
                checkpoint_path_override=args.checkpoint_path,
                attention_config_override=attention_config,
                offload=args.offload,
                magcache=args.magcache,
                quantized_qwen=args.qwen_quantization,
                attention_engine=args.attention_engine,
                blocks_in_memory=args.blocks_in_memory,
                enable_block_swap=True,
                dtype=model_dtype,
                use_mixed_weights=args.use_mixed_weights,
                text_encoder_dtype=text_encoder_dtype,
                vae_dtype=vae_dtype,
                computation_dtype=computation_dtype,
                use_int8=args.use_int8,
                int8_block_size=args.int8_block_size,
                use_fp8=use_fp8_computation,
                use_fp8_text_encoder=use_fp8_text_encoder,
                use_sdnq=use_sdnq,
                sdnq_weights_dtype=sdnq_weights_dtype,
                sdnq_use_quantized_matmul=sdnq_use_quantized_matmul,
                vae_temporal_tile_frames=args.vae_temporal_tile_frames,
                vae_temporal_stride_frames=args.vae_temporal_stride_frames,
                vae_spatial_tile_height=args.vae_spatial_tile_height,
                vae_spatial_tile_width=args.vae_spatial_tile_width,
            )
        else:
            # Use standard I2V pipeline
            pipe = get_I2V_pipeline(
                device_map={"dit": "cuda:0", "vae": "cuda:0",
                            "text_embedder": "cuda:0"},
                conf_path=args.config,
                checkpoint_path_override=args.checkpoint_path,
                attention_config_override=attention_config,
                offload=args.offload,
                magcache=args.magcache,
                quantized_qwen=args.qwen_quantization,
                attention_engine=args.attention_engine,
                dtype=model_dtype,
                use_mixed_weights=args.use_mixed_weights,
                text_encoder_dtype=text_encoder_dtype,
                vae_dtype=vae_dtype,
                computation_dtype=computation_dtype,
                use_int8=args.use_int8,
                int8_block_size=args.int8_block_size,
                use_fp8=use_fp8_computation,
                use_fp8_text_encoder=use_fp8_text_encoder,
                use_sdnq=use_sdnq,
                sdnq_weights_dtype=sdnq_weights_dtype,
                sdnq_use_quantized_matmul=sdnq_use_quantized_matmul,
                vae_temporal_tile_frames=args.vae_temporal_tile_frames,
                vae_temporal_stride_frames=args.vae_temporal_stride_frames,
                vae_spatial_tile_height=args.vae_spatial_tile_height,
                vae_spatial_tile_width=args.vae_spatial_tile_width,
            )
    else:  # T2V
        if is_t2v_pro and args.enable_block_swap:
            # Use block swapping pipeline for T2V Pro (20B model)
            pipe = get_T2V_pipeline_with_block_swap(
                device_map={"dit": "cuda:0", "vae": "cuda:0",
                            "text_embedder": "cuda:0"},
                resolution=512,
                conf_path=args.config,
                checkpoint_path_override=args.checkpoint_path,
                attention_config_override=attention_config,
                offload=args.offload,
                magcache=args.magcache,
                quantized_qwen=args.qwen_quantization,
                attention_engine=args.attention_engine,
                blocks_in_memory=args.blocks_in_memory,
                enable_block_swap=True,
                dtype=model_dtype,
                use_mixed_weights=args.use_mixed_weights,
                text_encoder_dtype=text_encoder_dtype,
                vae_dtype=vae_dtype,
                computation_dtype=computation_dtype,
                use_int8=args.use_int8,
                int8_block_size=args.int8_block_size,
                use_fp8=use_fp8_computation,
                use_fp8_text_encoder=use_fp8_text_encoder,
                use_sdnq=use_sdnq,
                sdnq_weights_dtype=sdnq_weights_dtype,
                sdnq_use_quantized_matmul=sdnq_use_quantized_matmul,
                vae_temporal_tile_frames=args.vae_temporal_tile_frames,
                vae_temporal_stride_frames=args.vae_temporal_stride_frames,
                vae_spatial_tile_height=args.vae_spatial_tile_height,
                vae_spatial_tile_width=args.vae_spatial_tile_width,
            )
        else:
            # Use standard T2V pipeline
            pipe = get_T2V_pipeline(
                device_map={"dit": "cuda:0", "vae": "cuda:0",
                            "text_embedder": "cuda:0"},
                conf_path=args.config,
                checkpoint_path_override=args.checkpoint_path,
                attention_config_override=attention_config,
                offload=args.offload,
                magcache=args.magcache,
                quantized_qwen=args.qwen_quantization,
                attention_engine=args.attention_engine,
                dtype=model_dtype,
                use_mixed_weights=args.use_mixed_weights,
                text_encoder_dtype=text_encoder_dtype,
                vae_dtype=vae_dtype,
                computation_dtype=computation_dtype,
                use_int8=args.use_int8,
                int8_block_size=args.int8_block_size,
                use_fp8=use_fp8_computation,
                use_fp8_text_encoder=use_fp8_text_encoder,
                use_sdnq=use_sdnq,
                sdnq_weights_dtype=sdnq_weights_dtype,
                sdnq_use_quantized_matmul=sdnq_use_quantized_matmul,
                vae_temporal_tile_frames=args.vae_temporal_tile_frames,
                vae_temporal_stride_frames=args.vae_temporal_stride_frames,
                vae_spatial_tile_height=args.vae_spatial_tile_height,
                vae_spatial_tile_width=args.vae_spatial_tile_width,
            )

    # Load LoRA adapters if specified
    if args.lora_path is not None and len(args.lora_path) > 0:
        print(f"\n>>> Loading {len(args.lora_path)} LoRA adapter(s)...")

        # Set default weights if not specified
        lora_weights = args.lora_weight if args.lora_weight else [1.0] * len(args.lora_path)
        if len(lora_weights) != len(args.lora_path):
            raise ValueError(f"Number of --lora_weight ({len(lora_weights)}) must match --lora_path ({len(args.lora_path)})")

        # Set default triggers if not specified
        lora_triggers = args.lora_trigger if args.lora_trigger else [None] * len(args.lora_path)
        if len(lora_triggers) != len(args.lora_path):
            raise ValueError(f"Number of --lora_trigger ({len(lora_triggers)}) must match --lora_path ({len(args.lora_path)})")

        for i, lora_item in enumerate(args.lora_path):
            # Detect LoRA format: folder (official PEFT) or single file (musubi tuner)
            if os.path.isdir(lora_item):
                # Official K5 LoRA format: folder with config_lora.json + lora.safetensors
                config_path = os.path.join(lora_item, "config_lora.json")
                weights_path = os.path.join(lora_item, "lora.safetensors")

                if not os.path.exists(config_path):
                    raise FileNotFoundError(f"LoRA config not found: {config_path}")
                if not os.path.exists(weights_path):
                    raise FileNotFoundError(f"LoRA weights not found: {weights_path}")

                adapter_name = os.path.basename(lora_item) or f"lora_{i}"

                print(f">>> Loading LoRA {i+1}/{len(args.lora_path)} (PEFT format): {lora_item}")
                print(f"    Adapter name: {adapter_name}, Weight: {lora_weights[i]}")

                pipe.load_adapter(
                    adapter_config=config_path,
                    adapter_path=weights_path,
                    adapter_name=adapter_name,
                    trigger=lora_triggers[i]
                )

                if lora_weights[i] != 1.0:
                    print(f"    Note: LoRA weight {lora_weights[i]} specified but PEFT adapter system uses full weight.")

            elif os.path.isfile(lora_item) and lora_item.endswith(".safetensors"):
                # Musubi tuner format: single .safetensors file
                print(f">>> Loading LoRA {i+1}/{len(args.lora_path)} (musubi format): {lora_item}")
                print(f"    Weight: {lora_weights[i]}")

                pipe.load_musubi_lora(
                    lora_path=lora_item,
                    multiplier=lora_weights[i],
                    trigger=lora_triggers[i]
                )
            else:
                raise ValueError(f"Invalid LoRA path: {lora_item}. Must be a directory or .safetensors file.")

        print(f">>> All LoRA adapters loaded successfully\n")

    if args.output_filename is None:
        # Determine file extension based on generation mode
        if is_t2i:
            ext = ".png"
        else:
            ext = ".mp4"
        args.output_filename = "./" + args.prompt.replace(" ", "_") + ext

    # Set up file-based signal checking for early stop
    stop_decode_file = args.output_filename + ".stop_decode"
    stop_save_file = args.output_filename + ".stop_save"
    # Checkpoint file handling for both image and video outputs
    if args.output_filename.endswith(".png"):
        checkpoint_file = args.output_filename.replace(".png", "_checkpoint.pt")
    else:
        checkpoint_file = args.output_filename.replace(".mp4", "_checkpoint.pt")

    def check_stop_signals():
        """Check for stop signal files and return action if found."""
        if os.path.exists(stop_decode_file):
            try:
                os.remove(stop_decode_file)
            except:
                pass
            return "decode"
        if os.path.exists(stop_save_file):
            try:
                os.remove(stop_save_file)
            except:
                pass
            return "save"
        return None

    start_time = time.perf_counter()

    # Handle decode from saved latents (VAE-only decoding)
    if args.decode_from_file:
        print(f">>> Decode mode: Loading latents from {args.decode_from_file}", flush=True)

        try:
            # Load latent checkpoint
            ckpt = torch.load(args.decode_from_file, map_location='cpu')

            latent_visual = ckpt["latents"]
            shape = ckpt["shape"]
            mode = ckpt.get("mode", "t2v")

            print(f">>> Latent shape: {latent_visual.shape}", flush=True)
            print(f">>> Mode: {mode}", flush=True)

            bs = shape[0]

            # Move VAE to device
            vae = pipe.vae.to("cuda")

            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=vae_dtype):
                    # Reshape latents: (frames, h, w, 16) -> (bs, frames, h, w, 16)
                    images = latent_visual.reshape(
                        bs,
                        -1,
                        latent_visual.shape[-3],
                        latent_visual.shape[-2],
                        latent_visual.shape[-1],
                    )
                    images = images.to(device="cuda")

                    # Scale and permute: (bs, frames, h, w, 16) -> (bs, 16, frames, h, w)
                    images = (images / vae.config.scaling_factor).permute(0, 4, 1, 2, 3)

                    # Decode
                    print(f">>> Decoding latents...", flush=True)
                    images = vae.decode(images).sample

                    # Convert to uint8
                    images = ((images.clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8)

            # Save the output
            if images is not None:
                import torchvision
                from torchvision.transforms import ToPILImage

                if mode == "t2i":
                    # Save as image
                    for image in images.squeeze(2).cpu():
                        pil_image = ToPILImage()(image)
                        pil_image.save(args.output_filename)
                    print(f"TIME ELAPSED: {time.perf_counter() - start_time}")
                    print(f"Decoded image saved to {args.output_filename}")
                else:
                    # Save as video
                    for video in images:
                        torchvision.io.write_video(
                            args.output_filename,
                            video.float().permute(1, 2, 3, 0).cpu().numpy(),
                            fps=24,
                            options={"crf": "5"},
                        )
                    print(f"TIME ELAPSED: {time.perf_counter() - start_time}")
                    print(f"Decoded video saved to {args.output_filename}")

        except Exception as e:
            print(f">>> ERROR during decode: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise

    # Handle resume from checkpoint
    elif args.resume_from:
        print(f">>> Resume mode: Loading checkpoint from {args.resume_from}", flush=True)

        try:
            # Load checkpoint to check mode
            ckpt = torch.load(args.resume_from, map_location='cpu')
            is_i2v_checkpoint = ckpt.get("mode") == "i2v" or ckpt.get("first_frames") is not None

            print(f">>> Checkpoint contains: step {ckpt.get('step')}/{ckpt.get('total_steps')}", flush=True)
            print(f">>> Mode: {'I2V' if is_i2v_checkpoint else 'T2V'}", flush=True)

            # Get DiT and VAE from the pipe (text embedder not needed for resume)
            force_offload = hasattr(pipe.dit, 'enable_block_swap') and pipe.dit.enable_block_swap

            if is_i2v_checkpoint:
                print(">>> Resuming I2V generation", flush=True)
                x = generate_sample_i2v_from_checkpoint(
                    checkpoint_path=args.resume_from,
                    dit=pipe.dit,
                    vae=pipe.vae,
                    conf=pipe.conf,
                    device="cuda",
                    vae_device="cuda",
                    progress=True,
                    offload=pipe.offload,
                    force_offload=force_offload,
                    stop_check=check_stop_signals,
                    new_checkpoint_path=checkpoint_file,
                )
            else:
                print(">>> Resuming T2V generation", flush=True)
                x = generate_sample_from_checkpoint(
                    checkpoint_path=args.resume_from,
                    dit=pipe.dit,
                    vae=pipe.vae,
                    conf=pipe.conf,
                    device="cuda",
                    vae_device="cuda",
                    progress=True,
                    offload=pipe.offload,
                    force_offload=force_offload,
                    stop_check=check_stop_signals,
                    new_checkpoint_path=checkpoint_file,
                )

            # Save the video if we got results
            if x is not None:
                import torchvision
                for video in x:
                    torchvision.io.write_video(
                        args.output_filename,
                        video.float().permute(1, 2, 3, 0).cpu().numpy(),
                        fps=24,
                        options={"crf": "5"},
                    )

            print(f"TIME ELAPSED: {time.perf_counter() - start_time}")
            if x is None:
                print(f">>> Checkpoint saved to {checkpoint_file}")
            else:
                print(f"Generated video is saved to {args.output_filename}")

        except Exception as e:
            print(f">>> ERROR during resume: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise

    elif is_t2i:
        # Text-to-Image generation
        x = pipe(args.prompt,
                 width=args.width,
                 height=args.height,
                 num_steps=args.sample_steps,
                 guidance_weight=args.guidance_weight,
                 scheduler_scale=args.scheduler_scale,
                 expand_prompts=args.expand_prompt,
                 save_path=args.output_filename,
                 seed=args.seed)
    elif args.denoise and args.video is not None:
        # VIDEO DENOISE MODE - applies light denoising to smooth artifacts
        print(f">>> VIDEO DENOISE MODE")
        print(f">>> Input video: {args.video}")
        print(f">>> Denoise strength: {args.denoise_strength}")

        alignment = 128 if args.attention_type == "nabla" else 32
        force_offload = hasattr(pipe.dit, 'enable_block_swap') and pipe.dit.enable_block_swap

        # Offload DiT BEFORE VAE encoding to free up VRAM
        if hasattr(pipe.dit, 'offload_all_blocks'):
            pipe.dit.offload_all_blocks()
        if pipe.offload or force_offload:
            pipe.dit = pipe.dit.to("cpu", non_blocking=True)
        torch.cuda.empty_cache()

        # Load VAE for encoding
        if pipe.offload or force_offload:
            pipe.vae = pipe.vae.to(pipe.device_map["vae"], non_blocking=True)

        # Encode video to latents (v2v_mode ensures edge-safe tiling)
        print(f">>> Encoding video to latent space...", flush=True)
        video_latents, scale_factor, num_frames = encode_video_to_latents(
            args.video,
            pipe.vae,
            pipe.device_map["vae"],
            alignment=alignment,
            v2v_mode=True,  # Use edge-safe tiling for v2v to prevent pixel loss
            target_width=args.width,
            target_height=args.height
        )

        # Offload VAE after encoding
        if pipe.offload or force_offload:
            pipe.vae = pipe.vae.to("cpu", non_blocking=True)
            torch.cuda.empty_cache()

        print(f">>> Video latents shape: {video_latents.shape}")
        print(f">>> Input video frames: {num_frames}, Latent frames: {video_latents.shape[0]}")

        # Load text embedder if needed
        if pipe.offload or force_offload:
            pipe.text_embedder = pipe.text_embedder.to(pipe.device_map["text_embedder"])

        # Apply denoising
        num_steps = args.sample_steps if args.sample_steps else pipe.num_steps
        guidance = args.guidance_weight if args.guidance_weight else min(pipe.guidance_weight, 4.0)

        # Calculate chunk_frames from video_duration setting
        # This ensures we respect --video_duration for section size
        chunk_frames = args.video_duration * 24 // 4 + 1  # 5s = 31 frames

        # Create previewer for DENOISE mode
        previewer = None
        if LatentPreviewer is not None and args.preview is not None and args.preview > 0:
            print(f"\n>>> DENOISE: Initializing previewer with preview={args.preview}")
            try:
                # For DENOISE mode, the previewer needs the NOISE (not clean latents)
                # to correctly subtract noise during preview generation.
                # Generate the same noise that will be used in generate_denoise using the same seed.
                # Must match: device, dtype, shape, and seed exactly for identical noise.
                g_preview = torch.Generator(device=pipe.device_map["dit"])
                g_preview.manual_seed(args.seed)
                initial_noise = torch.randn(
                    video_latents.shape,
                    device=pipe.device_map["dit"],
                    dtype=video_latents.dtype,  # Match the actual latents dtype
                    generator=g_preview
                )
                # Permute to match previewer expected format: [C, F, H, W]
                initial_latent = initial_noise.permute(3, 0, 1, 2).to(dtype=torch.bfloat16)

                timesteps = torch.linspace(args.denoise_strength, 0, num_steps + 1, device=pipe.device_map["dit"])
                timesteps = args.scheduler_scale * timesteps / (1 + (args.scheduler_scale - 1) * timesteps)
                timesteps = timesteps[:-1] * 1000

                class Args:
                    def __init__(self, save_path, fps):
                        self.save_path = save_path
                        self.fps = fps

                args_obj = Args(
                    save_path=os.path.dirname(args.output_filename) if args.output_filename else './',
                    fps=24
                )

                previewer = LatentPreviewer(
                    args=args_obj,
                    original_latents=initial_latent,
                    timesteps=timesteps,
                    device=pipe.device_map["dit"],
                    dtype=torch.bfloat16,
                    model_type="hunyuan"
                )
                print(f">>> DENOISE: Previewer initialized successfully, will generate preview every {args.preview} steps")
            except Exception as e:
                print(f">>> DENOISE: Failed to initialize previewer: {e}")
                import traceback
                traceback.print_exc()
                previewer = None
        else:
            print(f">>> DENOISE: Preview disabled (preview={args.preview})")

        x = generate_sample_denoise(
            video_latents=video_latents,
            caption=args.prompt,
            dit=pipe.dit,
            vae=pipe.vae,
            conf=pipe.conf,
            text_embedder=pipe.text_embedder,
            denoise_strength=args.denoise_strength,
            num_steps=num_steps,
            guidance_weight=guidance,
            scheduler_scale=args.scheduler_scale,
            negative_caption=args.negative_prompt,
            clip_prompt=args.clip_prompt,
            seed=args.seed,
            device=pipe.device_map["dit"],
            vae_device=pipe.device_map["vae"],
            progress=True,
            offload=pipe.offload,
            force_offload=force_offload,
            chunk_frames=chunk_frames,
            previewer=previewer,
            preview_interval=args.preview,
            preview_suffix=args.preview_suffix,
        )

        # Save output
        if x is not None:
            import torchvision
            torchvision.io.write_video(
                args.output_filename,
                x[0].float().permute(1, 2, 3, 0).cpu().numpy(),
                fps=24,
                options={"crf": "5"},
            )
            print(f">>> Saved denoised video to {args.output_filename}")

    elif is_i2v:
        if args.image is not None and args.end_image is not None and args.video is None:
            # ============================================================
            # IMAGE-TO-IMAGE-VIDEO MODE
            # Generate video transitioning from start image to end image
            # Uses same dual-conditioning as video join mode
            # ============================================================
            print(f">>> IMAGE-TO-IMAGE-VIDEO MODE")
            print(f">>> Start image: {args.image}")
            print(f">>> End image: {args.end_image}")

            alignment = 128 if args.attention_type == "nabla" else 32

            # Load VAE for encoding conditioning frames
            force_offload = hasattr(pipe.dit, 'enable_block_swap') and pipe.dit.enable_block_swap
            if pipe.offload or force_offload:
                pipe.vae = pipe.vae.to(pipe.device_map["vae"], non_blocking=True)

            # Optionally resize images to target dimensions
            start_image_path = args.image
            end_image_path = args.end_image
            if args.width and args.height:
                print(f">>> Resizing images to {args.width}x{args.height}")
                start_image_path = resize_image_to_resolution(args.image, args.width, args.height, alignment)
                end_image_path = resize_image_to_resolution(args.end_image, args.width, args.height, alignment)

            # Encode both images to latent space
            start_cond_latents, end_cond_latents, scale_factor = get_conditioning_latents_from_two_images(
                start_image_path,
                end_image_path,
                pipe.vae,
                pipe.device_map["vae"],
                alignment=alignment
            )

            # Offload VAE after encoding
            if pipe.offload or force_offload:
                pipe.vae = pipe.vae.to("cpu", non_blocking=True)
                torch.cuda.empty_cache()

            # Calculate output dimensions
            height, width = start_cond_latents.shape[1:3]
            total_frames = 1 if args.video_duration == 0 else args.video_duration * 24 // 4 + 1
            shape = (1, total_frames, height, width, 16)

            print(f">>> Start conditioning latents shape: {start_cond_latents.shape}")
            print(f">>> End conditioning latents shape: {end_cond_latents.shape}")
            print(f">>> Total frames (latent): {total_frames}")
            print(f">>> Output shape: {shape}")

            # Create previewer for I2I-Video mode
            previewer = None
            num_steps = args.sample_steps if args.sample_steps else pipe.num_steps
            if LatentPreviewer is not None and args.preview is not None and args.preview > 0:
                print(f"\n>>> I2I-Video: Initializing previewer with preview={args.preview}")
                try:
                    g_temp = torch.Generator(device=pipe.device_map["dit"])
                    g_temp.manual_seed(args.seed)
                    # Preview all frames being processed (start conditioning + middle + end conditioning)
                    initial_latent = torch.randn(shape[0] * total_frames, shape[2], shape[3], shape[4],
                                                device=pipe.device_map["dit"], generator=g_temp)
                    initial_latent = initial_latent.permute(3, 0, 1, 2)

                    timesteps = torch.linspace(1, 0, num_steps + 1, device=pipe.device_map["dit"])
                    timesteps = args.scheduler_scale * timesteps / (1 + (args.scheduler_scale - 1) * timesteps)
                    timesteps = timesteps[:-1] * 1000

                    class Args:
                        def __init__(self, save_path, fps):
                            self.save_path = save_path
                            self.fps = fps

                    args_obj = Args(
                        save_path=os.path.dirname(args.output_filename) if args.output_filename else './',
                        fps=24
                    )

                    previewer = LatentPreviewer(
                        args=args_obj,
                        original_latents=initial_latent,
                        timesteps=timesteps,
                        device=pipe.device_map["dit"],
                        dtype=torch.bfloat16,
                        model_type="hunyuan"
                    )
                    print(f">>> I2I-Video: Previewer initialized successfully")
                except Exception as e:
                    print(f">>> I2I-Video: Failed to initialize previewer: {e}")
                    import traceback
                    traceback.print_exc()
                    previewer = None

            # Generate video using dual-image conditioning
            # Reuses the same function as video join mode
            x = generate_sample_v2v_join(
                shape,
                args.prompt,
                pipe.dit,
                pipe.vae,
                pipe.conf,
                text_embedder=pipe.text_embedder,
                start_cond_latents=start_cond_latents,
                end_cond_latents=end_cond_latents,
                num_steps=num_steps,
                guidance_weight=args.guidance_weight if args.guidance_weight else pipe.guidance_weight,
                scheduler_scale=args.scheduler_scale,
                negative_caption=args.negative_prompt,
                clip_prompt=args.clip_prompt,
                seed=args.seed,
                device=pipe.device_map["dit"],
                vae_device=pipe.device_map["vae"],
                progress=True,
                offload=pipe.offload,
                force_offload=force_offload,
                previewer=previewer,
                preview_interval=args.preview,
                preview_suffix=args.preview_suffix,
                stop_check=check_stop_signals,
                checkpoint_path=checkpoint_file,
                save_latents=args.save_latents,
                use_apg=args.use_apg,
                apg_momentum=args.apg_momentum,
                apg_norm_threshold=args.apg_norm_threshold,
                end_blend_weight=args.end_blend_weight,
            )

            # Save output video directly (no concatenation needed unlike video join mode)
            if x is not None:
                import torchvision
                torchvision.io.write_video(
                    args.output_filename,
                    x[0].float().permute(1, 2, 3, 0).cpu().numpy(),
                    fps=24,
                    options={"crf": "5"},
                )
                print(f">>> Saved image-to-image video to {args.output_filename}")

        elif args.video is not None and args.video2 is not None:
            # Video-to-Video JOINING mode - create transition between two videos
            print(f">>> VIDEO JOINING MODE")
            print(f">>> First video: {args.video}")
            print(f">>> Second video: {args.video2}")
            print(f">>> Conditioning frames from each video: {args.num_cond_frames}")

            alignment = 128 if args.attention_type == "nabla" else 32

            # Load VAE for encoding conditioning frames
            force_offload = hasattr(pipe.dit, 'enable_block_swap') and pipe.dit.enable_block_swap
            if pipe.offload or force_offload:
                pipe.vae = pipe.vae.to(pipe.device_map["vae"], non_blocking=True)

            # Extract and encode conditioning frames from BOTH videos
            start_cond_latents, end_cond_latents, scale_factor = get_conditioning_frames_from_two_videos(
                args.video,
                args.video2,
                args.num_cond_frames,
                pipe.vae,
                pipe.device_map["vae"],
                alignment=alignment
            )

            # Offload VAE after encoding
            if pipe.offload or force_offload:
                pipe.vae = pipe.vae.to("cpu", non_blocking=True)
                torch.cuda.empty_cache()

            height, width = start_cond_latents.shape[1:3]
            # Total frames includes: start_cond + middle + end_cond
            # Calculate total frames based on desired video duration
            total_frames = 1 if args.video_duration == 0 else args.video_duration * 24 // 4 + 1
            shape = (1, total_frames, height, width, 16)

            print(f">>> Start conditioning latents shape: {start_cond_latents.shape}")
            print(f">>> End conditioning latents shape: {end_cond_latents.shape}")
            print(f">>> Total frames (model expects): {total_frames}")
            print(f">>> Output shape: {shape}")

            # Create previewer for V2V JOIN mode
            previewer = None
            num_steps = args.sample_steps if args.sample_steps else pipe.num_steps
            if LatentPreviewer is not None and args.preview is not None and args.preview > 0:
                print(f"\n>>> V2V JOIN: Initializing previewer with preview={args.preview}")
                try:
                    g_temp = torch.Generator(device=pipe.device_map["dit"])
                    g_temp.manual_seed(args.seed)
                    # Preview all frames being processed (start conditioning + middle + end conditioning)
                    initial_latent = torch.randn(shape[0] * total_frames, shape[2], shape[3], shape[4], device=pipe.device_map["dit"], generator=g_temp)
                    initial_latent = initial_latent.permute(3, 0, 1, 2)

                    timesteps = torch.linspace(1, 0, num_steps + 1, device=pipe.device_map["dit"])
                    timesteps = args.scheduler_scale * timesteps / (1 + (args.scheduler_scale - 1) * timesteps)
                    timesteps = timesteps[:-1] * 1000

                    class Args:
                        def __init__(self, save_path, fps):
                            self.save_path = save_path
                            self.fps = fps

                    args_obj = Args(
                        save_path=os.path.dirname(args.output_filename) if args.output_filename else './',
                        fps=24
                    )

                    previewer = LatentPreviewer(
                        args=args_obj,
                        original_latents=initial_latent,
                        timesteps=timesteps,
                        device=pipe.device_map["dit"],
                        dtype=torch.bfloat16,
                        model_type="hunyuan"
                    )
                    print(f">>> V2V JOIN: Previewer initialized successfully, will generate preview every {args.preview} steps")
                except Exception as e:
                    print(f">>> V2V JOIN: Failed to initialize previewer: {e}")
                    import traceback
                    traceback.print_exc()
                    previewer = None
            else:
                print(f">>> V2V JOIN: Preview disabled (preview={args.preview})")

            # Generate transition using dual conditioning
            x = generate_sample_v2v_join(
                shape,
                args.prompt,
                pipe.dit,
                pipe.vae,
                pipe.conf,
                text_embedder=pipe.text_embedder,
                start_cond_latents=start_cond_latents,
                end_cond_latents=end_cond_latents,
                num_steps=num_steps,
                guidance_weight=args.guidance_weight if args.guidance_weight else pipe.guidance_weight,
                scheduler_scale=args.scheduler_scale,
                negative_caption=args.negative_prompt,
                clip_prompt=args.clip_prompt,
                seed=args.seed,
                device=pipe.device_map["dit"],
                vae_device=pipe.device_map["vae"],
                progress=True,
                offload=pipe.offload,
                force_offload=force_offload,
                previewer=previewer,
                preview_interval=args.preview,
                preview_suffix=args.preview_suffix,
                stop_check=check_stop_signals,
                checkpoint_path=checkpoint_file,
                save_latents=args.save_latents,
                use_apg=args.use_apg,
                apg_momentum=args.apg_momentum,
                apg_norm_threshold=args.apg_norm_threshold,
                end_blend_weight=args.end_blend_weight,
            )

            # Concatenate: video1 + generated middle + video2
            if x is not None:
                import torchvision
                import av
                import numpy as np

                # Load BOTH input videos at 24 fps
                print(f">>> Loading first video for concatenation: {args.video}")
                container1 = av.open(args.video)
                video_stream1 = container1.streams.video[0]
                video_fps1 = float(video_stream1.average_rate)

                # Calculate frame skip for 24 fps target
                if video_fps1 > 24:
                    frame_skip1 = int(video_fps1 / 24)
                else:
                    frame_skip1 = 1

                input_frames1 = []
                frame_count = 0
                for frame in container1.decode(video=0):
                    if frame_count % frame_skip1 == 0:
                        img = frame.to_ndarray(format='rgb24')
                        input_frames1.append(img)
                    frame_count += 1
                container1.close()

                print(f">>> Loading second video for concatenation: {args.video2}")
                container2 = av.open(args.video2)
                video_stream2 = container2.streams.video[0]
                video_fps2 = float(video_stream2.average_rate)

                if video_fps2 > 24:
                    frame_skip2 = int(video_fps2 / 24)
                else:
                    frame_skip2 = 1

                input_frames2 = []
                frame_count = 0
                for frame in container2.decode(video=0):
                    if frame_count % frame_skip2 == 0:
                        img = frame.to_ndarray(format='rgb24')
                        input_frames2.append(img)
                    frame_count += 1
                container2.close()

                # Convert to tensors
                input_video1_tensor = torch.from_numpy(np.stack(input_frames1)).float()
                input_video2_tensor = torch.from_numpy(np.stack(input_frames2)).float()

                # Get the generated video (includes start, middle, end)
                # x shape: [1, C, total_frames, H, W] -> [total_frames, H, W, C]
                generated_video = x[0].float().permute(1, 2, 3, 0).cpu()

                # Extract only the MIDDLE section (exclude conditioning frames)
                # VAE has ~4x temporal compression: video_frames = 1 + (latent_frames - 1) * 4
                num_cond_video_frames = 1 + (args.num_cond_frames - 1) * 4
                middle_frames = generated_video[num_cond_video_frames:-num_cond_video_frames]

                print(f">>> Conditioning: {args.num_cond_frames} latent frames = {num_cond_video_frames} video frames each side")
                print(f">>> Generated middle frames: {middle_frames.shape[0]}")

                # Resize videos to match generated resolution if needed
                gen_h, gen_w = middle_frames.shape[1:3]
                if input_video1_tensor.shape[1] != gen_h or input_video1_tensor.shape[2] != gen_w:
                    print(f">>> Resizing first video from {input_video1_tensor.shape[1]}x{input_video1_tensor.shape[2]} to {gen_h}x{gen_w}")
                    input_video1_tensor = torch.nn.functional.interpolate(
                        input_video1_tensor.permute(0, 3, 1, 2),
                        size=(gen_h, gen_w),
                        mode='bilinear',
                        align_corners=False
                    ).permute(0, 2, 3, 1)

                if input_video2_tensor.shape[1] != gen_h or input_video2_tensor.shape[2] != gen_w:
                    print(f">>> Resizing second video from {input_video2_tensor.shape[1]}x{input_video2_tensor.shape[2]} to {gen_h}x{gen_w}")
                    input_video2_tensor = torch.nn.functional.interpolate(
                        input_video2_tensor.permute(0, 3, 1, 2),
                        size=(gen_h, gen_w),
                        mode='bilinear',
                        align_corners=False
                    ).permute(0, 2, 3, 1)

                # Apply frame normalization if enabled
                if args.normalize_frames and args.normalize_frames > 0:
                    print(f">>> Applying frame normalization with {args.normalize_frames} frames at each join point")
                    input_video1_tensor, middle_frames, input_video2_tensor = normalize_join_frames_triple(
                        input_video1_tensor, middle_frames, input_video2_tensor, args.normalize_frames
                    )

                # Concatenate: full video1 + middle + full video2
                final_video = torch.cat([input_video1_tensor, middle_frames, input_video2_tensor], dim=0)

                print(f">>> First video frames: {input_video1_tensor.shape[0]}")
                print(f">>> Middle generated frames: {middle_frames.shape[0]}")
                print(f">>> Second video frames: {input_video2_tensor.shape[0]}")
                print(f">>> Final joined video frames: {final_video.shape[0]}")

                torchvision.io.write_video(
                    args.output_filename,
                    final_video.numpy(),
                    fps=24,
                    options={"crf": "5"},
                )
                print(f">>> Saved joined video to {args.output_filename}")

        elif args.video is not None and args.end_image is not None:
            # Video + End Image mode - create transition from video to ending image
            print(f">>> VIDEO + END IMAGE MODE")
            print(f">>> Input video: {args.video}")
            print(f">>> End image: {args.end_image}")
            print(f">>> Conditioning frames from video: {args.num_cond_frames}")

            alignment = 128 if args.attention_type == "nabla" else 32

            # Load VAE for encoding conditioning frames
            force_offload = hasattr(pipe.dit, 'enable_block_swap') and pipe.dit.enable_block_swap
            if pipe.offload or force_offload:
                pipe.vae = pipe.vae.to(pipe.device_map["vae"], non_blocking=True)

            # Extract and encode conditioning frames from video and end image
            start_cond_latents, end_cond_latents, scale_factor = get_conditioning_video_and_image(
                args.video,
                args.end_image,
                args.num_cond_frames,
                pipe.vae,
                pipe.device_map["vae"],
                alignment=alignment
            )

            # Offload VAE after encoding
            if pipe.offload or force_offload:
                pipe.vae = pipe.vae.to("cpu", non_blocking=True)
                torch.cuda.empty_cache()

            height, width = start_cond_latents.shape[1:3]
            total_frames = 1 if args.video_duration == 0 else args.video_duration * 24 // 4 + 1
            shape = (1, total_frames, height, width, 16)

            print(f">>> Start conditioning latents shape: {start_cond_latents.shape}")
            print(f">>> End conditioning latents shape: {end_cond_latents.shape}")
            print(f">>> Total frames (latent): {total_frames}")
            print(f">>> Output shape: {shape}")

            # Create previewer for Video + End Image mode
            previewer = None
            num_steps = args.sample_steps if args.sample_steps else pipe.num_steps
            if LatentPreviewer is not None and args.preview is not None and args.preview > 0:
                print(f"\n>>> Video+EndImage: Initializing previewer with preview={args.preview}")
                try:
                    g_temp = torch.Generator(device=pipe.device_map["dit"])
                    g_temp.manual_seed(args.seed)
                    initial_latent = torch.randn(shape[0] * total_frames, shape[2], shape[3], shape[4],
                                                device=pipe.device_map["dit"], generator=g_temp)
                    initial_latent = initial_latent.permute(3, 0, 1, 2)

                    timesteps = torch.linspace(1, 0, num_steps + 1, device=pipe.device_map["dit"])
                    timesteps = args.scheduler_scale * timesteps / (1 + (args.scheduler_scale - 1) * timesteps)
                    timesteps = timesteps[:-1] * 1000

                    class Args:
                        def __init__(self, save_path, fps):
                            self.save_path = save_path
                            self.fps = fps

                    args_obj = Args(
                        save_path=os.path.dirname(args.output_filename) if args.output_filename else './',
                        fps=24
                    )

                    previewer = LatentPreviewer(
                        args=args_obj,
                        original_latents=initial_latent,
                        timesteps=timesteps,
                        device=pipe.device_map["dit"],
                        dtype=torch.bfloat16,
                        model_type="hunyuan"
                    )
                    print(f">>> Video+EndImage: Previewer initialized successfully")
                except Exception as e:
                    print(f">>> Video+EndImage: Failed to initialize previewer: {e}")
                    import traceback
                    traceback.print_exc()
                    previewer = None

            # Generate video using dual conditioning (video start + image end)
            x = generate_sample_v2v_join(
                shape,
                args.prompt,
                pipe.dit,
                pipe.vae,
                pipe.conf,
                text_embedder=pipe.text_embedder,
                start_cond_latents=start_cond_latents,
                end_cond_latents=end_cond_latents,
                num_steps=num_steps,
                guidance_weight=args.guidance_weight if args.guidance_weight else pipe.guidance_weight,
                scheduler_scale=args.scheduler_scale,
                negative_caption=args.negative_prompt,
                clip_prompt=args.clip_prompt,
                seed=args.seed,
                device=pipe.device_map["dit"],
                vae_device=pipe.device_map["vae"],
                progress=True,
                offload=pipe.offload,
                force_offload=force_offload,
                previewer=previewer,
                preview_interval=args.preview,
                preview_suffix=args.preview_suffix,
                stop_check=check_stop_signals,
                checkpoint_path=checkpoint_file,
                save_latents=args.save_latents,
                use_apg=args.use_apg,
                apg_momentum=args.apg_momentum,
                apg_norm_threshold=args.apg_norm_threshold,
                end_blend_weight=args.end_blend_weight,
            )

            # Concatenate input video with generated frames
            if x is not None:
                import torchvision
                import av

                # Load original input video frames
                print(f">>> Loading input video for concatenation: {args.video}")
                container = av.open(args.video)
                stream = container.streams.video[0]
                input_frames = []
                for frame in container.decode(video=0):
                    input_frames.append(frame.to_ndarray(format='rgb24'))
                container.close()

                input_video_tensor = torch.from_numpy(np.stack(input_frames)).float()
                print(f">>> Input video: {input_video_tensor.shape[0]} frames at {input_video_tensor.shape[1]}x{input_video_tensor.shape[2]}")

                # Get generated middle frames (excluding conditioning frames at both ends)
                generated_video = x[0].float().permute(1, 2, 3, 0).cpu()
                gen_h, gen_w = generated_video.shape[1], generated_video.shape[2]
                num_cond = args.num_cond_frames
                middle_frames = generated_video[num_cond:-num_cond] if num_cond > 0 else generated_video
                print(f">>> Generated middle frames: {middle_frames.shape[0]} frames")

                # Resize input video if needed to match generated resolution
                if input_video_tensor.shape[1] != gen_h or input_video_tensor.shape[2] != gen_w:
                    print(f">>> Resizing input video from {input_video_tensor.shape[1]}x{input_video_tensor.shape[2]} to {gen_h}x{gen_w}")
                    input_video_tensor = torch.nn.functional.interpolate(
                        input_video_tensor.permute(0, 3, 1, 2),
                        size=(gen_h, gen_w),
                        mode='bilinear',
                        align_corners=False
                    ).permute(0, 2, 3, 1)

                # Normalize if requested
                if args.normalize_frames and args.normalize_frames > 0:
                    input_video_tensor, middle_frames = normalize_join_frames(
                        input_video_tensor, middle_frames, args.normalize_frames
                    )

                # Concatenate: full input video + middle frames (no second video to append)
                final_video = torch.cat([input_video_tensor, middle_frames], dim=0)
                print(f">>> Final video: {final_video.shape[0]} frames")
                print(f">>> Input video frames: {input_video_tensor.shape[0]}")
                print(f">>> Generated middle frames: {middle_frames.shape[0]}")

                torchvision.io.write_video(
                    args.output_filename,
                    final_video.numpy(),
                    fps=24,
                    options={"crf": "5"},
                )
                print(f">>> Saved video + end image result to {args.output_filename}")

        elif args.video is not None:
            # Video-to-Video continuation mode
            print(f">>> VIDEO CONTINUATION MODE")
            print(f">>> Input video: {args.video}")
            print(f">>> Conditioning frames: {args.num_cond_frames}")

            alignment = 128 if args.attention_type == "nabla" else 32

            # Load VAE for encoding conditioning frames
            force_offload = hasattr(pipe.dit, 'enable_block_swap') and pipe.dit.enable_block_swap
            if pipe.offload or force_offload:
                pipe.vae = pipe.vae.to(pipe.device_map["vae"], non_blocking=True)

            # Extract and encode conditioning frames from video
            cond_latents, scale_factor = get_conditioning_frames_from_video(
                args.video,
                args.num_cond_frames,
                pipe.vae,
                pipe.device_map["vae"],
                alignment=alignment
            )

            # Offload VAE after encoding
            if pipe.offload or force_offload:
                pipe.vae = pipe.vae.to("cpu", non_blocking=True)
                torch.cuda.empty_cache()

            height, width = cond_latents.shape[1:3]
            # Calculate new frames: total frames - conditioning frames
            # This ensures total output matches model's trained frame count
            total_frames = 1 if args.video_duration == 0 else args.video_duration * 24 // 4 + 1
            num_new_frames = total_frames - args.num_cond_frames
            shape = (1, num_new_frames, height, width, 16)

            print(f">>> Conditioning latents shape: {cond_latents.shape}")
            print(f">>> Total frames (model expects): {total_frames}")
            print(f">>> New frames to generate: {num_new_frames}")
            print(f">>> Output shape: {shape}")

            # Create previewer for V2V mode
            previewer = None
            num_steps = args.sample_steps if args.sample_steps else pipe.num_steps
            if LatentPreviewer is not None and args.preview is not None and args.preview > 0:
                print(f"\n>>> V2V: Initializing previewer with preview={args.preview}")
                try:
                    g_temp = torch.Generator(device=pipe.device_map["dit"])
                    g_temp.manual_seed(args.seed)
                    # Preview all frames being processed (conditioning + new)
                    initial_latent = torch.randn(shape[0] * total_frames, shape[2], shape[3], shape[4], device=pipe.device_map["dit"], generator=g_temp)
                    initial_latent = initial_latent.permute(3, 0, 1, 2)

                    timesteps = torch.linspace(1, 0, num_steps + 1, device=pipe.device_map["dit"])
                    timesteps = args.scheduler_scale * timesteps / (1 + (args.scheduler_scale - 1) * timesteps)
                    timesteps = timesteps[:-1] * 1000

                    class Args:
                        def __init__(self, save_path, fps):
                            self.save_path = save_path
                            self.fps = fps

                    args_obj = Args(
                        save_path=os.path.dirname(args.output_filename) if args.output_filename else './',
                        fps=24
                    )

                    previewer = LatentPreviewer(
                        args=args_obj,
                        original_latents=initial_latent,
                        timesteps=timesteps,
                        device=pipe.device_map["dit"],
                        dtype=torch.bfloat16,
                        model_type="hunyuan"
                    )
                    print(f">>> V2V: Previewer initialized successfully, will generate preview every {args.preview} steps")
                except Exception as e:
                    print(f">>> V2V: Failed to initialize previewer: {e}")
                    import traceback
                    traceback.print_exc()
                    previewer = None
            else:
                print(f">>> V2V: Preview disabled (preview={args.preview})")

            # Generate continuation using visual conditioning (I2V approach)
            x = generate_sample_v2v(
                shape,
                args.prompt,
                pipe.dit,
                pipe.vae,
                pipe.conf,
                text_embedder=pipe.text_embedder,
                cond_latents=cond_latents,
                num_steps=num_steps,
                guidance_weight=args.guidance_weight if args.guidance_weight else pipe.guidance_weight,
                scheduler_scale=args.scheduler_scale,
                negative_caption=args.negative_prompt,
                clip_prompt=args.clip_prompt,
                seed=args.seed,
                device=pipe.device_map["dit"],
                vae_device=pipe.device_map["vae"],
                progress=True,
                offload=pipe.offload,
                force_offload=force_offload,
                previewer=previewer,
                preview_interval=args.preview,
                preview_suffix=args.preview_suffix,
                stop_check=check_stop_signals,
                checkpoint_path=checkpoint_file,
                save_latents=args.save_latents,
                use_apg=args.use_apg,
                apg_momentum=args.apg_momentum,
                apg_norm_threshold=args.apg_norm_threshold,
            )

            # Concatenate input video with newly generated frames
            if x is not None:
                import torchvision
                import av
                import numpy as np

                # Load original input video frames at 24 fps
                print(f">>> Loading input video for concatenation: {args.video}")
                container = av.open(args.video)
                video_stream = container.streams.video[0]
                video_fps = float(video_stream.average_rate)

                # Calculate frame skip for 24 fps target
                if video_fps > 24:
                    frame_skip = int(video_fps / 24)
                else:
                    frame_skip = 1

                input_frames = []
                frame_count = 0
                for frame in container.decode(video=0):
                    if frame_count % frame_skip == 0:
                        img = frame.to_ndarray(format='rgb24')
                        input_frames.append(img)
                    frame_count += 1
                container.close()

                # Convert to tensor [num_frames, H, W, C]
                input_video_tensor = torch.from_numpy(np.stack(input_frames)).float()

                # Get only the NEW generated frames (exclude conditioning frames)
                # x shape: [1, C, total_frames, H, W] -> [total_frames, H, W, C]
                generated_video = x[0].float().permute(1, 2, 3, 0).cpu()

                # Convert latent conditioning frames to video frames
                # VAE has ~4x temporal compression: video_frames = 1 + (latent_frames - 1) * 4
                num_cond_video_frames = 1 + (args.num_cond_frames - 1) * 4
                new_frames = generated_video[num_cond_video_frames:]  # Exclude conditioning frames

                print(f">>> Conditioning: {args.num_cond_frames} latent frames = {num_cond_video_frames} video frames")

                # Resize input frames to match generated resolution if needed
                gen_h, gen_w = new_frames.shape[1:3]
                if input_video_tensor.shape[1] != gen_h or input_video_tensor.shape[2] != gen_w:
                    print(f">>> Resizing input video from {input_video_tensor.shape[1]}x{input_video_tensor.shape[2]} to {gen_h}x{gen_w}")
                    input_video_tensor = torch.nn.functional.interpolate(
                        input_video_tensor.permute(0, 3, 1, 2),  # [N, C, H, W]
                        size=(gen_h, gen_w),
                        mode='bilinear',
                        align_corners=False
                    ).permute(0, 2, 3, 1)  # [N, H, W, C]

                # Apply frame normalization if enabled
                if args.normalize_frames and args.normalize_frames > 0:
                    print(f">>> Applying frame normalization with {args.normalize_frames} frames at join point")
                    input_video_tensor, new_frames = normalize_join_frames(
                        input_video_tensor, new_frames, args.normalize_frames
                    )

                # Concatenate: input video + new generated frames
                final_video = torch.cat([input_video_tensor, new_frames], dim=0)

                print(f">>> Input video frames: {input_video_tensor.shape[0]}")
                print(f">>> New generated frames: {new_frames.shape[0]}")
                print(f">>> Final video frames: {final_video.shape[0]}")

                torchvision.io.write_video(
                    args.output_filename,
                    final_video.numpy(),
                    fps=24,
                    options={"crf": "5"},
                )
                print(f">>> Saved concatenated video to {args.output_filename}")
        else:
            # Standard Image-to-Video mode
            image_to_use = args.image
            if args.width and args.height:
                alignment = 128 if args.attention_type == "nabla" else 32
                print(f"Resizing input image to {args.width}x{args.height} for i2v mode (alignment: {alignment})")
                image_to_use = resize_image_to_resolution(args.image, args.width, args.height, alignment)

            x = pipe(args.prompt,
                     image=image_to_use,
                     time_length=args.video_duration,
                     num_steps=args.sample_steps,
                     guidance_weight=args.guidance_weight,
                     scheduler_scale=args.scheduler_scale,
                     negative_caption=args.negative_prompt,
                     expand_prompts=args.expand_prompt,
                     clip_prompt=args.clip_prompt,
                     save_path=args.output_filename,
                     seed=args.seed,
                     preview=args.preview,
                     preview_suffix=args.preview_suffix,
                     stop_check=check_stop_signals,
                     checkpoint_path=checkpoint_file,
                     save_latents=args.save_latents)
    else:
        x = pipe(args.prompt,
             time_length=args.video_duration,
             width=args.width,
             height=args.height,
             num_steps=args.sample_steps,
             guidance_weight=args.guidance_weight,
             scheduler_scale=args.scheduler_scale,
             negative_caption=args.negative_prompt,
             expand_prompts=args.expand_prompt,
             clip_prompt=args.clip_prompt,
             save_path=args.output_filename,
             seed=args.seed,
             preview=args.preview,
             preview_suffix=args.preview_suffix,
             stop_check=check_stop_signals,
             checkpoint_path=checkpoint_file,
             save_latents=args.save_latents)

    print(f"TIME ELAPSED: {time.perf_counter() - start_time}")

    if x is None:
        print(f">>> Checkpoint saved to {checkpoint_file}")
        print(f">>> No output generated (latents saved for later)")
    else:
        output_type = "image" if is_t2i else "video"
        print(f"Generated {output_type} is saved to {args.output_filename}")
    