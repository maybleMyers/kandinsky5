import argparse
import time
import warnings
import logging
import os
import tempfile
import sys

# --- EARLY CONFIGURATION START ---
# 1. Mock triton on Windows if cl.exe is missing to prevent bitsandbytes/diffusers crash
if sys.platform == "win32":
    import shutil
    if shutil.which("cl.exe") is None:
        from unittest.mock import MagicMock
        from importlib.machinery import ModuleSpec
        import types
        
        # Create a mock module for triton
        triton_mock = MagicMock()
        triton_mock.__spec__ = ModuleSpec(name="triton", loader=None)
        triton_mock.__path__ = [] # Essential to make it behave like a package
        sys.modules["triton"] = triton_mock
        
        # Create a mock module for triton.language
        triton_language_mock = MagicMock()
        triton_language_mock.__spec__ = ModuleSpec(name="triton.language", loader=None)
        sys.modules["triton.language"] = triton_language_mock
        triton_mock.language = triton_language_mock

        # Create a mock module for triton.compiler
        triton_compiler_mock = MagicMock()
        triton_compiler_mock.__spec__ = ModuleSpec(name="triton.compiler", loader=None)
        sys.modules["triton.compiler"] = triton_compiler_mock
        triton_mock.compiler = triton_compiler_mock
        
        # Add CompiledKernel to triton.compiler
        triton_compiler_mock.CompiledKernel = MagicMock()
        
        print("WARNING: Triton mocked to bypass missing cl.exe (C++ compiler).")

# 2. Early parse --no_compile to set environment variable BEFORE importing kandinsky
def _early_parse_no_compile():
    for i, arg in enumerate(sys.argv):
        if arg == '--no_compile':
            return True
    return False

_no_compile = _early_parse_no_compile()
if _no_compile:
    os.environ["KANDINSKY_NO_COMPILE"] = "1"
    print("torch.compile() disabled via environment variable for faster startup")
# --- EARLY CONFIGURATION END ---

import torch
from PIL import Image

# Monkeypatch torch.compile to be a no-op if --no_compile is set
# This prevents bitsandbytes and other libraries from triggering compilation
if _no_compile:
    print("Overriding torch.compile to be a no-op...")
    def no_op_compile(model=None, **kwargs):
        if model is None:
            def decorator(func):
                return func
            return decorator
        return model
    torch.compile = no_op_compile

# Now it is safe to import kandinsky modules
import kandinsky.models.compile_config as compile_config
# Double check that the flag was set correctly
if _no_compile and compile_config.USE_TORCH_COMPILE:
    print("WARNING: compile_config.USE_TORCH_COMPILE is still True despite --no_compile!")
    compile_config.USE_TORCH_COMPILE = False # Force it just in case

from kandinsky import get_T2V_pipeline, get_I2V_pipeline, get_I2V_pipeline_with_block_swap, get_T2V_pipeline_with_block_swap, get_T2I_pipeline
from kandinsky.generation_utils import generate_sample_from_checkpoint, generate_sample_i2v_from_checkpoint

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
        help="The prompt to generate video"
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

    # INT8 quantization configuration
    parser.add_argument(
        "--use_int8",
        action='store_true',
        default=False,
        help="Use INT8 quantization for linear layers (40-60%% memory reduction, 1.5-2x faster inference)"
    )
    parser.add_argument(
        "--int8_block_size",
        type=int,
        default=128,
        help="Block size for INT8 quantization (must be 128 for Triton kernels, default: 128)"
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
                vae_temporal_tile_frames=args.vae_temporal_tile_frames,
                vae_temporal_stride_frames=args.vae_temporal_stride_frames,
                vae_spatial_tile_height=args.vae_spatial_tile_height,
                vae_spatial_tile_width=args.vae_spatial_tile_width,
            )

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

    # Handle resume from checkpoint
    if args.resume_from:
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
    elif is_i2v:
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
                 expand_prompts=args.expand_prompt,
                 clip_prompt=args.clip_prompt,
                 save_path=args.output_filename,
                 seed=args.seed,
                 preview=args.preview,
                 preview_suffix=args.preview_suffix,
                 stop_check=check_stop_signals,
                 checkpoint_path=checkpoint_file)
    else:
        x = pipe(args.prompt,
             time_length=args.video_duration,
             width=args.width,
             height=args.height,
             num_steps=args.sample_steps,
             guidance_weight=args.guidance_weight,
             scheduler_scale=args.scheduler_scale,
             expand_prompts=args.expand_prompt,
             clip_prompt=args.clip_prompt,
             save_path=args.output_filename,
             seed=args.seed,
             preview=args.preview,
             preview_suffix=args.preview_suffix,
             stop_check=check_stop_signals,
             checkpoint_path=checkpoint_file)

    print(f"TIME ELAPSED: {time.perf_counter() - start_time}")

    if x is None:
        print(f">>> Checkpoint saved to {checkpoint_file}")
        print(f">>> No output generated (latents saved for later)")
    else:
        output_type = "image" if is_t2i else "video"
        print(f"Generated {output_type} is saved to {args.output_filename}")
