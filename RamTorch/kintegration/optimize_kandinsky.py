#!/usr/bin/env python3
"""
Complete Kandinsky INT8 Optimization Script

This script provides end-to-end optimization of Kandinsky models using
Triton INT8 quantization for significant memory reduction and speedup.
"""

import argparse
import time
import json
import torch
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
import warnings

# Try to import Kandinsky components
try:
    from kandinsky import get_T2V_pipeline, get_I2V_pipeline
    KANDINSKY_AVAILABLE = True
except ImportError:
    KANDINSKY_AVAILABLE = False
    warnings.warn("Kandinsky not available. Install with: pip install kandinsky")

# Import our optimization modules
try:
    from kandinsky_int8_optimizer import KandinskyInt8Optimizer
    from triton_int8_converter import DiffusionModelInt8Converter, TritonInt8Config
except ImportError:
    print("Error: Required modules not found. Ensure kandinsky_int8_optimizer.py and triton_int8_converter.py are in the same directory.")
    sys.exit(1)


def print_header(title: str, width: int = 80):
    """Print a formatted header"""
    print("\n" + "="*width)
    print(title.center(width))
    print("="*width)


def analyze_model(model_path: str, config_path: str) -> Dict:
    """
    Analyze a Kandinsky model for INT8 conversion potential.
    
    Args:
        model_path: Path to the model checkpoint
        config_path: Path to the model config file
        
    Returns:
        Dictionary with analysis results
    """
    print_header("MODEL ANALYSIS")
    
    print(f"\nModel path: {model_path}")
    print(f"Config path: {config_path}")
    
    # Initialize optimizer
    optimizer = KandinskyInt8Optimizer(
        model_path=model_path,
        block_size=128  # Required for Triton kernels
    )
    
    # Run analysis
    print("\nAnalyzing model structure...")
    analysis = optimizer.analyze_model()
    
    # Print summary
    print("\n" + "-"*80)
    print("ANALYSIS RESULTS")
    print("-"*80)
    
    total = analysis['total_params']
    int8_eligible = analysis['int8_eligible']
    fp32_required = analysis['fp32_required']
    
    if total > 0:
        print(f"\nTotal parameters: {total:,}")
        print(f"INT8 eligible: {int8_eligible:,} ({int8_eligible/total*100:.1f}%)")
        print(f"FP32/BF16 required: {fp32_required:,} ({fp32_required/total*100:.1f}%)")
    
    mem = analysis.get('memory_estimation', {})
    if mem:
        print(f"\nMemory Estimation:")
        print(f"  Current size: {mem.get('original_gb', 0):.2f} GB")
        print(f"  After optimization: {mem.get('optimized_gb', 0):.2f} GB")
        print(f"  Reduction: {mem.get('reduction_gb', 0):.2f} GB ({mem.get('reduction_pct', 0):.1f}%)")
    
    # Show top layers for INT8
    print("\n" + "-"*80)
    print("TOP LAYERS FOR INT8 QUANTIZATION")
    print("-"*80)
    
    int8_layers = [
        (name, info) for name, info in analysis['layer_analysis'].items()
        if info['recommended_precision'] == 'int8'
    ]
    
    # Sort by parameter count
    int8_layers.sort(key=lambda x: x[1]['params'], reverse=True)
    
    print("\nLargest INT8-eligible layers:")
    for i, (name, info) in enumerate(int8_layers[:10]):
        size_mb = (info['params'] * 4) / (1024**2)  # Original size in MB
        saved_mb = size_mb * 0.75  # 75% reduction for INT8
        print(f"  {i+1}. {name}")
        print(f"     Shape: {info['shape']}, Size: {size_mb:.1f}MB → {size_mb-saved_mb:.1f}MB")
    
    if len(int8_layers) > 10:
        print(f"\n  ... and {len(int8_layers)-10} more layers")
    
    return analysis


def optimize_model(
    model_path: str,
    output_path: str,
    analysis: Dict,
    use_triton: bool = True
) -> None:
    """
    Optimize the model with INT8 quantization.
    
    Args:
        model_path: Path to the input model
        output_path: Path for the optimized model
        analysis: Analysis results from analyze_model
        use_triton: Whether to use Triton kernels
    """
    print_header("MODEL OPTIMIZATION")
    
    print(f"\nInput: {model_path}")
    print(f"Output: {output_path}")
    print(f"Using Triton: {use_triton}")
    
    # Create output directory
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize converter with Triton config
    config = TritonInt8Config(block_size=128)
    
    # Adjust config based on analysis
    if analysis['int8_eligible'] / analysis['total_params'] < 0.3:
        print("\nNote: Low INT8 eligibility detected. Adjusting thresholds...")
        config.max_outlier_ratio = 0.02  # Allow more outliers
        config.min_elements_for_int8 = 2048  # Lower size threshold
    
    # Create converter
    converter = DiffusionModelInt8Converter(
        model_path=model_path,
        output_path=str(output_path),
        config=config
    )
    
    # Run conversion
    print("\nConverting model layers...")
    converter.convert()
    
    print(f"\n✓ Model optimized successfully!")
    print(f"✓ Saved to: {output_path}")


def benchmark_inference(
    original_path: str,
    optimized_path: str,
    config_path: str,
    prompt: str = "A majestic dragon soaring through clouds",
    duration: int = 2,
    resolution: Tuple[int, int] = (512, 512),
    steps: int = 10
) -> Dict:
    """
    Benchmark original vs optimized model.
    
    Args:
        original_path: Path to original model
        optimized_path: Path to optimized model  
        config_path: Path to config file
        prompt: Test prompt for generation
        duration: Video duration in seconds
        resolution: Width and height tuple
        steps: Number of diffusion steps
        
    Returns:
        Dictionary with benchmark results
    """
    if not KANDINSKY_AVAILABLE:
        print("\n⚠ Kandinsky not available. Skipping benchmark.")
        return {}
    
    print_header("PERFORMANCE BENCHMARK")
    
    print(f"\nTest configuration:")
    print(f"  Prompt: {prompt}")
    print(f"  Duration: {duration}s")
    print(f"  Resolution: {resolution[0]}x{resolution[1]}")
    print(f"  Steps: {steps}")
    
    results = {}
    
    # Benchmark original model
    print("\n" + "-"*80)
    print("ORIGINAL MODEL")
    print("-"*80)
    
    # Load model
    print("\nLoading original model...")
    torch.cuda.empty_cache()
    start_mem = torch.cuda.memory_allocated() / 1024**3
    
    pipe_original = get_T2V_pipeline(
        conf_path=config_path,
        checkpoint_path_override=original_path,
        dtype=torch.float32,
        device_map={"dit": "cuda:0", "vae": "cuda:0", "text_embedder": "cuda:0"}
    )
    
    loaded_mem = torch.cuda.memory_allocated() / 1024**3
    original_mem = loaded_mem - start_mem
    print(f"Memory used: {original_mem:.2f} GB")
    
    # Warm-up
    print("Warming up...")
    _ = pipe_original(
        prompt,
        time_length=1,
        width=256,
        height=256,
        num_steps=2
    )
    
    # Benchmark
    print(f"Running benchmark ({steps} steps)...")
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    _ = pipe_original(
        prompt,
        time_length=duration,
        width=resolution[0],
        height=resolution[1],
        num_steps=steps
    )
    
    torch.cuda.synchronize()
    original_time = time.perf_counter() - start_time
    
    results['original'] = {
        'memory_gb': original_mem,
        'time_seconds': original_time,
        'steps': steps
    }
    
    print(f"Time: {original_time:.2f}s")
    print(f"Speed: {steps/original_time:.2f} steps/s")
    
    # Clean up
    del pipe_original
    torch.cuda.empty_cache()
    
    # Benchmark optimized model
    print("\n" + "-"*80)
    print("OPTIMIZED MODEL (INT8)")
    print("-"*80)
    
    # Load model
    print("\nLoading optimized model...")
    start_mem = torch.cuda.memory_allocated() / 1024**3
    
    pipe_optimized = get_T2V_pipeline(
        conf_path=config_path,
        checkpoint_path_override=optimized_path,
        dtype=torch.bfloat16,
        use_mixed_weights=True,
        device_map={"dit": "cuda:0", "vae": "cuda:0", "text_embedder": "cuda:0"}
    )
    
    loaded_mem = torch.cuda.memory_allocated() / 1024**3
    optimized_mem = loaded_mem - start_mem
    print(f"Memory used: {optimized_mem:.2f} GB")
    
    # Warm-up
    print("Warming up...")
    _ = pipe_optimized(
        prompt,
        time_length=1,
        width=256,
        height=256,
        num_steps=2
    )
    
    # Benchmark
    print(f"Running benchmark ({steps} steps)...")
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    _ = pipe_optimized(
        prompt,
        time_length=duration,
        width=resolution[0],
        height=resolution[1],
        num_steps=steps
    )
    
    torch.cuda.synchronize()
    optimized_time = time.perf_counter() - start_time
    
    results['optimized'] = {
        'memory_gb': optimized_mem,
        'time_seconds': optimized_time,
        'steps': steps
    }
    
    print(f"Time: {optimized_time:.2f}s")
    print(f"Speed: {steps/optimized_time:.2f} steps/s")
    
    # Clean up
    del pipe_optimized
    torch.cuda.empty_cache()
    
    # Print comparison
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    
    mem_reduction = (1 - optimized_mem/original_mem) * 100
    speedup = original_time / optimized_time
    
    print(f"\nMemory Usage:")
    print(f"  Original: {original_mem:.2f} GB")
    print(f"  Optimized: {optimized_mem:.2f} GB")
    print(f"  Reduction: {mem_reduction:.1f}%")
    
    print(f"\nInference Speed:")
    print(f"  Original: {original_time:.2f}s ({steps/original_time:.2f} steps/s)")
    print(f"  Optimized: {optimized_time:.2f}s ({steps/optimized_time:.2f} steps/s)")
    print(f"  Speedup: {speedup:.2f}x")
    
    results['comparison'] = {
        'memory_reduction_pct': mem_reduction,
        'speedup_factor': speedup
    }
    
    return results


def verify_quality(
    original_path: str,
    optimized_path: str,
    config_path: str,
    num_samples: int = 3
) -> None:
    """
    Generate samples to verify quality preservation.
    
    Args:
        original_path: Path to original model
        optimized_path: Path to optimized model
        config_path: Path to config file
        num_samples: Number of samples to generate
    """
    if not KANDINSKY_AVAILABLE:
        print("\n⚠ Kandinsky not available. Skipping quality verification.")
        return
    
    print_header("QUALITY VERIFICATION")
    
    prompts = [
        "A serene lake reflecting mountains at sunset",
        "Cyberpunk city with neon lights in the rain",
        "Ancient tree in a mystical forest with fireflies"
    ][:num_samples]
    
    print(f"\nGenerating {num_samples} samples with each model...")
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n[{i}/{num_samples}] {prompt}")
        
        # Generate with original
        print("  Original model...", end="", flush=True)
        pipe_orig = get_T2V_pipeline(
            conf_path=config_path,
            checkpoint_path_override=original_path,
            dtype=torch.float32
        )
        
        orig_video = pipe_orig(
            prompt,
            time_length=2,
            width=512,
            height=512,
            num_steps=20,
            save_path=f"original_sample_{i}.mp4"
        )
        print(" ✓")
        
        del pipe_orig
        torch.cuda.empty_cache()
        
        # Generate with optimized
        print("  Optimized model...", end="", flush=True)
        pipe_opt = get_T2V_pipeline(
            conf_path=config_path,
            checkpoint_path_override=optimized_path,
            dtype=torch.bfloat16,
            use_mixed_weights=True
        )
        
        opt_video = pipe_opt(
            prompt,
            time_length=2,
            width=512,
            height=512,
            num_steps=20,
            save_path=f"optimized_sample_{i}.mp4"
        )
        print(" ✓")
        
        del pipe_opt
        torch.cuda.empty_cache()
    
    print(f"\n✓ Generated {num_samples} comparison samples")
    print("  Review the videos to verify quality preservation")


def save_report(
    output_path: str,
    analysis: Dict,
    benchmark: Dict
) -> None:
    """Save optimization report to JSON"""
    
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'model_analysis': analysis,
        'benchmark_results': benchmark,
        'optimization_config': {
            'block_size': 128,
            'quantization': 'triton_int8',
            'fallback_precision': 'bfloat16'
        }
    }
    
    report_path = Path(output_path) / 'optimization_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Optimize Kandinsky models with Triton INT8 quantization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic optimization
  python optimize_kandinsky.py --model-path ./models/kandinsky_t2v --config-path ./configs/config.yaml

  # With custom output and benchmarking
  python optimize_kandinsky.py --model-path ./models/kandinsky_t2v --config-path ./configs/config.yaml \\
      --output-path ./models/kandinsky_int8 --benchmark

  # Full pipeline with quality verification
  python optimize_kandinsky.py --model-path ./models/kandinsky_t2v --config-path ./configs/config.yaml \\
      --benchmark --verify-quality --num-samples 3
        """
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to Kandinsky model checkpoint directory"
    )
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path to model config file (e.g., config_5s.yaml)"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output path for optimized model (default: model-path_int8)"
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze model, don't optimize"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmark after optimization"
    )
    parser.add_argument(
        "--verify-quality",
        action="store_true",
        help="Generate sample videos to verify quality"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of samples for quality verification (default: 3)"
    )
    parser.add_argument(
        "--benchmark-steps",
        type=int,
        default=10,
        help="Number of diffusion steps for benchmark (default: 10)"
    )
    parser.add_argument(
        "--benchmark-resolution",
        type=int,
        nargs=2,
        default=[512, 512],
        help="Resolution for benchmark as width height (default: 512 512)"
    )
    parser.add_argument(
        "--no-triton",
        action="store_true",
        help="Disable Triton kernels (use fallback quantization)"
    )
    
    args = parser.parse_args()
    
    # Set default output path
    if args.output_path is None:
        model_name = Path(args.model_path).name
        args.output_path = str(Path(args.model_path).parent / f"{model_name}_int8")
    
    print_header("KANDINSKY INT8 OPTIMIZATION PIPELINE", 80)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model_path}")
    print(f"  Config: {args.config_path}")
    print(f"  Output: {args.output_path}")
    print(f"  Triton: {'Disabled' if args.no_triton else 'Enabled'}")
    
    # Step 1: Analyze model
    analysis = analyze_model(args.model_path, args.config_path)
    
    if args.analyze_only:
        print("\n[Analyze-only mode] Skipping optimization")
        return
    
    # Step 2: Optimize model
    print("\nProceed with optimization? [y/N]: ", end="")
    if input().lower() != 'y':
        print("Optimization cancelled")
        return
    
    optimize_model(
        args.model_path,
        args.output_path,
        analysis,
        use_triton=not args.no_triton
    )
    
    # Step 3: Benchmark (optional)
    benchmark_results = {}
    if args.benchmark:
        benchmark_results = benchmark_inference(
            args.model_path,
            args.output_path,
            args.config_path,
            resolution=tuple(args.benchmark_resolution),
            steps=args.benchmark_steps
        )
    
    # Step 4: Verify quality (optional)
    if args.verify_quality:
        verify_quality(
            args.model_path,
            args.output_path,
            args.config_path,
            num_samples=args.num_samples
        )
    
    # Step 5: Save report
    save_report(args.output_path, analysis, benchmark_results)
    
    print_header("OPTIMIZATION COMPLETE", 80)
    print(f"\n✓ Optimized model saved to: {args.output_path}")
    
    if benchmark_results:
        comp = benchmark_results.get('comparison', {})
        print(f"✓ Memory reduction: {comp.get('memory_reduction_pct', 0):.1f}%")
        print(f"✓ Speed improvement: {comp.get('speedup_factor', 1):.2f}x")
    
    print("\nNext steps:")
    print("1. Test the optimized model with your pipeline")
    print("2. Review generated samples if --verify-quality was used")
    print("3. Check optimization_report.json for detailed metrics")


if __name__ == "__main__":
    main()
