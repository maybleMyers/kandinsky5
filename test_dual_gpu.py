"""
Dual GPU test script for Kandinsky 5 with block swapping.

Supports two modes:
1. Pipeline Parallelism (--pipeline): Split diffusion steps between GPUs for faster single video
2. Parallel Generation: Run two separate videos on two GPUs simultaneously

For a 19B model with ~60 blocks:
- Each GPU keeps `blocks_in_memory` blocks loaded (e.g., 4-6)
- Remaining blocks are swapped to CPU as needed
"""

import argparse
import time
import warnings
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch

from kandinsky.utils import set_hf_token
from kandinsky import get_I2V_pipeline_with_block_swap, get_T2V_pipeline_with_block_swap, get_T2V_pipeline_dual_gpu


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate videos using Kandinsky 5 with dual GPU block swapping"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/config_5s_t2v_pro_20b.yaml",
        help="The config file of the model"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The dragon breathes fire in a medieval castle.",
        help="The prompt to generate video"
    )
    parser.add_argument(
        "--prompt2",
        type=str,
        default=None,
        help="Second prompt for parallel generation on GPU 1 (optional)"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="./assets/test_image.jpg",
        help="An image to generate video from (for I2V mode)"
    )
    parser.add_argument(
        "--image2",
        type=str,
        default=None,
        help="Second image for parallel generation on GPU 1 (optional)"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="Static, 2D cartoon, cartoon, 2d animation, paintings, images, worst quality, low quality, ugly, deformed, walking backwards",
        help="Negative prompt for classifier-free guidance"
    )
    parser.add_argument(
        "--video_duration",
        type=int,
        default=5,
        help="Duration of the video in seconds"
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
        default=None,
        help="Name of the resulting file"
    )
    parser.add_argument(
        "--output_filename2",
        type=str,
        default=None,
        help="Name of the resulting file (GPU 1 output for parallel mode)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1137,
        help="Seed for the random number generator"
    )
    parser.add_argument(
        "--seed2",
        type=int,
        default=None,
        help="Seed for GPU 1 (defaults to seed+1)"
    )
    parser.add_argument(
        "--blocks_in_memory",
        type=int,
        default=4,
        help="Number of transformer blocks to keep in GPU memory per GPU"
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
        help="Name of the full attention algorithm to use",
        choices=["flash_attention_2", "flash_attention_3", "sdpa", "sage", "auto"]
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Token to download restricted models",
    )
    parser.add_argument(
        "--gpu0",
        type=int,
        default=0,
        help="First GPU device ID"
    )
    parser.add_argument(
        "--gpu1",
        type=int,
        default=1,
        help="Second GPU device ID"
    )
    parser.add_argument(
        "--pipeline",
        action='store_true',
        default=False,
        help="Use pipeline parallelism (split steps between GPUs for faster single video)"
    )
    parser.add_argument(
        "--parallel",
        action='store_true',
        default=False,
        help="Run two separate videos in parallel (requires --prompt2)"
    )
    parser.add_argument(
        "--single_gpu",
        type=int,
        default=None,
        help="Use only a single GPU (specify 0 or 1)"
    )
    args = parser.parse_args()

    if args.hf_token:
        set_hf_token(args.hf_token)

    return args


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_pipeline_for_gpu(gpu_id, args, is_i2v=True):
    """Create a pipeline with block swapping for a specific GPU."""
    device_map = {
        "dit": torch.device(f'cuda:{gpu_id}'),
        "vae": torch.device(f'cuda:{gpu_id}'),
        "text_embedder": torch.device(f'cuda:{gpu_id}')
    }

    print(f"\n{'='*60}")
    print(f"Loading model on GPU {gpu_id} with block swapping...")
    print(f"  Blocks in memory: {args.blocks_in_memory}")
    print(f"  Quantized Qwen: {args.qwen_quantization}")
    print(f"{'='*60}\n")

    if is_i2v:
        pipe = get_I2V_pipeline_with_block_swap(
            device_map=device_map,
            conf_path=args.config,
            offload=True,
            quantized_qwen=args.qwen_quantization,
            attention_engine=args.attention_engine,
            blocks_in_memory=args.blocks_in_memory,
            enable_block_swap=True,
        )
    else:
        pipe = get_T2V_pipeline_with_block_swap(
            device_map=device_map,
            conf_path=args.config,
            offload=True,
            quantized_qwen=args.qwen_quantization,
            attention_engine=args.attention_engine,
            blocks_in_memory=args.blocks_in_memory,
            enable_block_swap=True,
        )

    return pipe


def generate_on_gpu(pipe, prompt, image, args, output_path, seed, gpu_id):
    """Run generation on a specific GPU."""
    print(f"\n[GPU {gpu_id}] Starting generation...")
    print(f"[GPU {gpu_id}] Prompt: {prompt[:50]}...")
    print(f"[GPU {gpu_id}] Seed: {seed}")

    start_time = time.perf_counter()

    torch.cuda.set_device(gpu_id)

    # Check if I2V or T2V based on whether image is provided
    if image is not None:
        result = pipe(
            prompt,
            image=image,
            time_length=args.video_duration,
            num_steps=args.sample_steps,
            guidance_weight=args.guidance_weight,
            scheduler_scale=args.scheduler_scale,
            expand_prompts=args.expand_prompt,
            negative_caption=args.negative_prompt,
            save_path=output_path,
            seed=seed,
        )
    else:
        result = pipe(
            prompt,
            time_length=args.video_duration,
            num_steps=args.sample_steps,
            guidance_weight=args.guidance_weight,
            scheduler_scale=args.scheduler_scale,
            expand_prompts=args.expand_prompt,
            negative_caption=args.negative_prompt,
            save_path=output_path,
            seed=seed,
        )

    elapsed = time.perf_counter() - start_time
    print(f"\n[GPU {gpu_id}] Generation complete in {elapsed:.1f}s")
    print(f"[GPU {gpu_id}] Saved to: {output_path}")

    return result, elapsed


def run_pipeline_parallel(args):
    """Run pipeline parallelism mode - split steps between two GPUs."""
    print(f"\n{'='*60}")
    print(f"Pipeline Parallel Mode")
    print(f"  Splitting diffusion steps between GPU {args.gpu0} and GPU {args.gpu1}")
    print(f"  Config: {args.config}")
    print(f"  Blocks in memory per GPU: {args.blocks_in_memory}")
    print(f"{'='*60}\n")

    device0 = torch.device(f'cuda:{args.gpu0}')
    device1 = torch.device(f'cuda:{args.gpu1}')

    # Create dual GPU pipeline
    pipe = get_T2V_pipeline_dual_gpu(
        device0=device0,
        device1=device1,
        conf_path=args.config,
        quantized_qwen=args.qwen_quantization,
        attention_engine=args.attention_engine,
        blocks_in_memory=args.blocks_in_memory,
    )

    # Generate video
    print(f"\nGenerating video...")
    print(f"  Prompt: {args.prompt[:80]}...")
    print(f"  Seed: {args.seed}")

    start_time = time.perf_counter()

    result = pipe(
        text=args.prompt,
        time_length=args.video_duration,
        num_steps=args.sample_steps,
        guidance_weight=args.guidance_weight,
        scheduler_scale=args.scheduler_scale,
        expand_prompts=bool(args.expand_prompt),
        negative_caption=args.negative_prompt,
        save_path=args.output_filename,
        seed=args.seed,
    )

    total_time = time.perf_counter() - start_time

    print(f"\n{'='*60}")
    print(f"Pipeline Parallel Generation Complete!")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Output: {args.output_filename}")
    print(f"{'='*60}\n")

    return result


def main():
    disable_warnings()
    args = parse_args()

    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"\nDetected {num_gpus} GPU(s)")

    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        free_mem, total_mem = torch.cuda.mem_get_info(i)
        print(f"  GPU {i}: {props.name} - {total_mem/1024**3:.1f}GB total, {free_mem/1024**3:.1f}GB free")

    # Determine if this is i2v or t2v
    is_i2v = "i2v" in args.config

    # Set default output filenames
    if args.output_filename is None:
        base_name = args.prompt.replace(" ", "_")[:32]
        if args.pipeline:
            args.output_filename = f"./{base_name}_pipeline.mp4"
        else:
            args.output_filename = f"./{base_name}_gpu0.mp4"

    if args.output_filename2 is None and args.prompt2:
        base_name = args.prompt2.replace(" ", "_")[:32]
        args.output_filename2 = f"./{base_name}_gpu1.mp4"

    # Set default seed2
    if args.seed2 is None:
        args.seed2 = args.seed + 1

    # Pipeline parallelism mode (recommended for faster single video)
    if args.pipeline:
        if is_i2v:
            print("\nNote: Pipeline parallelism currently only supports T2V mode.")
            print("Please use a T2V config (e.g., config_5s_t2v_pro_20b.yaml)")
            return

        if num_gpus < 2:
            print("\nError: Pipeline mode requires at least 2 GPUs")
            return

        set_seed(args.seed)
        run_pipeline_parallel(args)
        return

    # Single GPU mode
    if args.single_gpu is not None:
        gpu_id = args.single_gpu
        print(f"\n=== Single GPU Mode (GPU {gpu_id}) ===")

        set_seed(args.seed)
        pipe = create_pipeline_for_gpu(gpu_id, args, is_i2v)

        image = args.image if is_i2v else None
        generate_on_gpu(pipe, args.prompt, image, args, args.output_filename, args.seed, gpu_id)
        return

    # Dual GPU mode
    if num_gpus < 2:
        print("\nError: Dual GPU mode requires at least 2 GPUs")
        print("Use --single_gpu 0 to run on a single GPU")
        return

    # Parallel execution on both GPUs (two separate videos)
    if args.parallel:
        if not args.prompt2:
            print("\nError: Parallel mode requires --prompt2")
            return

        print(f"\n=== Parallel Dual GPU Mode ===")
        print(f"GPU {args.gpu0}: {args.prompt[:40]}...")
        print(f"GPU {args.gpu1}: {args.prompt2[:40]}...")

        print("\nLoading models on both GPUs (this may take a while)...")

        pipe0 = create_pipeline_for_gpu(args.gpu0, args, is_i2v)
        pipe1 = create_pipeline_for_gpu(args.gpu1, args, is_i2v)

        image0 = args.image if is_i2v else None
        image1 = args.image2 if args.image2 else args.image
        if is_i2v and image1 is None:
            image1 = args.image

        start_time = time.perf_counter()

        with ThreadPoolExecutor(max_workers=2) as executor:
            future0 = executor.submit(
                generate_on_gpu, pipe0, args.prompt, image0, args,
                args.output_filename, args.seed, args.gpu0
            )
            future1 = executor.submit(
                generate_on_gpu, pipe1, args.prompt2, image1, args,
                args.output_filename2, args.seed2, args.gpu1
            )

            results = []
            for future in as_completed([future0, future1]):
                try:
                    result, elapsed = future.result()
                    results.append((result, elapsed))
                except Exception as e:
                    print(f"Generation failed: {e}")
                    import traceback
                    traceback.print_exc()

        total_time = time.perf_counter() - start_time
        print(f"\n{'='*60}")
        print(f"Both generations complete!")
        print(f"Total wall time: {total_time:.1f}s")
        print(f"Output 1: {args.output_filename}")
        print(f"Output 2: {args.output_filename2}")
        print(f"{'='*60}")

    else:
        # Default: Sequential execution on GPU 0
        print(f"\n=== Sequential Mode (GPU {args.gpu0}) ===")
        print("Tip: Use --pipeline for faster single video generation using both GPUs")

        set_seed(args.seed)

        pipe0 = create_pipeline_for_gpu(args.gpu0, args, is_i2v)
        image = args.image if is_i2v else None
        generate_on_gpu(pipe0, args.prompt, image, args, args.output_filename, args.seed, args.gpu0)


if __name__ == "__main__":
    main()
