#!/usr/bin/env python3
"""
Kandinsky INT8 Quantization Analysis Script

This script analyzes your Kandinsky model to determine:
- How many parameters can be quantized to INT8
- Expected memory savings
- Which layers will be quantized
- Compatibility check for block size requirements

Usage:
    python scripts/analyze_kandinsky_int8.py --config ./configs/config_5s_sft.yaml
    python scripts/analyze_kandinsky_int8.py --config ./configs/config_pro_20b.yaml --strategy aggressive
"""

import argparse
import sys
from pathlib import Path
import torch
import yaml

# Add kandinsky to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kandinsky.models.dit import get_dit
from kandinsky.models.int8_utils import (
    Int8Config,
    analyze_model_for_int8,
    print_int8_analysis,
    get_quantizable_layers,
    check_layer_compatibility
)


def load_config(config_path: str) -> dict:
    """Load model configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Kandinsky model for INT8 quantization potential",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze T2V 5s model
  python scripts/analyze_kandinsky_int8.py --config ./configs/config_5s_sft.yaml

  # Analyze Pro 20B model with aggressive strategy
  python scripts/analyze_kandinsky_int8.py --config ./configs/config_pro_20b.yaml --strategy aggressive

  # Analyze I2V model
  python scripts/analyze_kandinsky_int8.py --config ./configs/config_i2v.yaml
        """
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to model config file (e.g., ./configs/config_5s_sft.yaml)"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="auto",
        choices=["auto", "aggressive", "conservative"],
        help="INT8 quantization strategy (default: auto)"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed per-layer analysis"
    )
    parser.add_argument(
        "--check-compatibility",
        action="store_true",
        help="Check layer-by-layer compatibility with INT8"
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("KANDINSKY INT8 QUANTIZATION ANALYSIS")
    print("=" * 80)
    print(f"\nConfig: {args.config}")
    print(f"Strategy: {args.strategy}")

    # Load config
    print("\nLoading model configuration...")
    config = load_config(args.config)

    # Extract DiT config
    dit_config = config.get('diffusion_transformer', {})

    print(f"Model parameters:")
    print(f"  - Model dim: {dit_config.get('model_dim', 2048)}")
    print(f"  - FF dim: {dit_config.get('ff_dim', 5120)}")
    print(f"  - Text blocks: {dit_config.get('num_text_blocks', 2)}")
    print(f"  - Visual blocks: {dit_config.get('num_visual_blocks', 32)}")

    # Create INT8 config
    int8_config = Int8Config(
        enabled=True,
        strategy=args.strategy,
        block_size=128
    )

    # Build model (on CPU to avoid VRAM usage)
    print("\nBuilding model (this may take a moment)...")
    dit_config_with_int8 = dit_config.copy()
    dit_config_with_int8['use_int8'] = False  # Build without INT8 first for analysis

    try:
        with torch.no_grad():
            model = get_dit(dit_config_with_int8)
            model.eval()

        print("Model built successfully!")

        # Analyze model
        print("\nAnalyzing model layers...")
        analysis = analyze_model_for_int8(model, int8_config)

        # Print analysis
        print_int8_analysis(analysis)

        # Detailed layer listing
        if args.detailed:
            print("\n" + "=" * 80)
            print("DETAILED LAYER ANALYSIS")
            print("=" * 80)

            quantizable_layers = get_quantizable_layers(model, int8_config)

            print(f"\nQuantizable layers ({len(quantizable_layers)}):")
            for i, (name, layer) in enumerate(quantizable_layers.items(), 1):
                params = layer.in_features * layer.out_features
                memory_fp32_mb = (params * 4) / (1024**2)
                memory_int8_mb = (params * 1) / (1024**2)
                savings_mb = memory_fp32_mb - memory_int8_mb

                print(f"\n{i}. {name}")
                print(f"   Shape: [{layer.out_features}, {layer.in_features}]")
                print(f"   Params: {params:,}")
                print(f"   Memory: {memory_fp32_mb:.2f} MB → {memory_int8_mb:.2f} MB (saves {savings_mb:.2f} MB)")

        # Compatibility check
        if args.check_compatibility:
            print("\n" + "=" * 80)
            print("COMPATIBILITY CHECK")
            print("=" * 80)

            print("\nChecking all linear layers for INT8 compatibility...")

            incompatible_layers = []
            compatible_count = 0

            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    is_compatible, reason = check_layer_compatibility(module, block_size=128)

                    if not is_compatible:
                        incompatible_layers.append((name, reason))
                    else:
                        compatible_count += 1

            print(f"\nCompatible layers: {compatible_count}")
            print(f"Incompatible layers: {len(incompatible_layers)}")

            if incompatible_layers:
                print("\nIncompatible layers (will use BF16/FP32):")
                for i, (name, reason) in enumerate(incompatible_layers[:20], 1):
                    print(f"  {i}. {name}")
                    print(f"     Reason: {reason}")

                if len(incompatible_layers) > 20:
                    print(f"  ... and {len(incompatible_layers) - 20} more")

        # Provide recommendations
        print("\n" + "=" * 80)
        print("RECOMMENDATIONS")
        print("=" * 80)

        savings_pct = analysis['memory_savings_pct']
        quantizable_pct = analysis['quantizable_pct']

        if savings_pct > 40:
            print("\n✓ EXCELLENT: INT8 quantization is highly recommended!")
            print(f"  Expected memory reduction: {savings_pct:.1f}%")
            print(f"  This will significantly improve inference efficiency.")
        elif savings_pct > 25:
            print("\n✓ GOOD: INT8 quantization will provide noticeable benefits")
            print(f"  Expected memory reduction: {savings_pct:.1f}%")
        else:
            print("\n⚠ MODERATE: INT8 quantization will provide limited benefits")
            print(f"  Expected memory reduction: {savings_pct:.1f}%")

        if quantizable_pct < 50:
            print("\nNote: Less than 50% of parameters can be quantized.")
            print("This is normal for models with many embedding/normalization layers.")

        print("\nNext steps:")
        print("1. Run test.py with --use_int8 flag to test INT8 inference:")
        print(f"   python test.py --config {args.config} --use_int8")
        print("\n2. Compare quality and speed with/without INT8")
        print("\n3. For best results, use INT8 with BF16 computation dtype:")
        print(f"   python test.py --config {args.config} --use_int8 --dtype bfloat16")

    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
