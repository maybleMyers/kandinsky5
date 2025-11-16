import argparse
import time
import warnings
import logging
import os
import tempfile

import torch
from PIL import Image

from kandinsky import get_T2V_pipeline, get_I2V_pipeline, get_I2V_pipeline_with_block_swap, get_T2V_pipeline_with_block_swap



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


def resize_image_to_resolution(image_path, target_width, target_height):
    """
    Resize image to target resolution while maintaining aspect ratio and ensuring
    dimensions are multiples of 32.

    Args:
        image_path: Path to the input image
        target_width: Target width (should be multiple of 32)
        target_height: Target height (should be multiple of 32)

    Returns:
        Path to the resized image (temporary file)
    """
    try:
        img = Image.open(image_path)
        original_width, original_height = img.size

        # Ensure target dimensions are multiples of 32
        target_width = (target_width // 32) * 32
        target_height = (target_height // 32) * 32
        target_width = max(64, target_width)
        target_height = max(64, target_height)

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
        choices=["float32", "float16", "bfloat16"],
        help="Data type for model weights (default: bfloat16). Use bfloat16 for best memory efficiency with minimal quality loss. This sets all dtypes if specific ones are not provided."
    )
    parser.add_argument(
        "--text_encoder_dtype",
        type=str,
        default=None,
        choices=["float32", "float16", "bfloat16"],
        help="Data type specifically for text encoder. If not set, uses --dtype value."
    )
    parser.add_argument(
        "--vae_dtype",
        type=str,
        default=None,
        choices=["float32", "float16", "bfloat16"],
        help="Data type specifically for VAE. If not set, uses --dtype value."
    )
    parser.add_argument(
        "--computation_dtype",
        type=str,
        default=None,
        choices=["float32", "float16", "bfloat16"],
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
    }
    model_dtype = dtype_map[args.dtype]

    # Set individual component dtypes (fall back to model_dtype if not specified)
    text_encoder_dtype = dtype_map[args.text_encoder_dtype] if args.text_encoder_dtype else model_dtype
    vae_dtype = dtype_map[args.vae_dtype] if args.vae_dtype else model_dtype
    computation_dtype = dtype_map[args.computation_dtype] if args.computation_dtype else model_dtype

    # Determine model type from config filename
    is_i2v = "i2v" in args.config.lower()
    is_t2v_pro = "t2v" in args.config.lower() and ("pro" in args.config.lower() or "20b" in args.config.lower())

    if is_i2v:
        if args.enable_block_swap:
            # Use block swapping pipeline for large I2V models
            pipe = get_I2V_pipeline_with_block_swap(
                device_map={"dit": "cuda:0", "vae": "cuda:0",
                            "text_embedder": "cuda:0"},
                conf_path=args.config,
                checkpoint_path_override=args.checkpoint_path,
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
            )
        else:
            # Use standard I2V pipeline
            pipe = get_I2V_pipeline(
                device_map={"dit": "cuda:0", "vae": "cuda:0",
                            "text_embedder": "cuda:0"},
                conf_path=args.config,
                checkpoint_path_override=args.checkpoint_path,
                offload=args.offload,
                magcache=args.magcache,
                quantized_qwen=args.qwen_quantization,
                attention_engine=args.attention_engine,
                dtype=model_dtype,
                use_mixed_weights=args.use_mixed_weights,
                text_encoder_dtype=text_encoder_dtype,
                vae_dtype=vae_dtype,
                computation_dtype=computation_dtype,
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
            )
        else:
            # Use standard T2V pipeline
            pipe = get_T2V_pipeline(
                device_map={"dit": "cuda:0", "vae": "cuda:0",
                            "text_embedder": "cuda:0"},
                conf_path=args.config,
                checkpoint_path_override=args.checkpoint_path,
                offload=args.offload,
                magcache=args.magcache,
                quantized_qwen=args.qwen_quantization,
                attention_engine=args.attention_engine,
                dtype=model_dtype,
                use_mixed_weights=args.use_mixed_weights,
                text_encoder_dtype=text_encoder_dtype,
                vae_dtype=vae_dtype,
                computation_dtype=computation_dtype,
            )

    if args.output_filename is None:
        args.output_filename = "./" + args.prompt.replace(" ", "_") + ".mp4"

    start_time = time.perf_counter()
    if is_i2v:
        # Resize image if width and height are specified
        image_to_use = args.image
        if args.width and args.height:
            print(f"Resizing input image to {args.width}x{args.height} for i2v mode")
            image_to_use = resize_image_to_resolution(args.image, args.width, args.height)

        x = pipe(args.prompt,
                 image=image_to_use,
                 time_length=args.video_duration,
                 num_steps=args.sample_steps,
                 guidance_weight=args.guidance_weight,
                 scheduler_scale=args.scheduler_scale,
                 expand_prompts=args.expand_prompt,
                 save_path=args.output_filename,
                 seed=args.seed)
    else:  # T2V
        x = pipe(args.prompt,
             time_length=args.video_duration,
             width=args.width,
             height=args.height,
             num_steps=args.sample_steps,
             guidance_weight=args.guidance_weight,
             scheduler_scale=args.scheduler_scale,
             expand_prompts=args.expand_prompt,
             save_path=args.output_filename,
             seed=args.seed)
    print(f"TIME ELAPSED: {time.perf_counter() - start_time}")
    print(f"Generated video is saved to {args.output_filename}")
    