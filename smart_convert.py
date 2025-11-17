#!/usr/bin/env python3
"""
Mixed Precision Model Converter

Converts a sharded diffusers model to mixed precision based on analysis results,
keeping critical weights in FP32 while converting safe weights to BF16/FP16.
"""

import os
import json
import argparse
import shutil
from pathlib import Path
from collections import defaultdict
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm


class MixedPrecisionConverter:
    def __init__(self, model_path, analysis_path, target_precision='bf16'):
        self.model_path = Path(model_path)
        self.analysis_path = Path(analysis_path)
        self.target_precision = target_precision
        self.analysis = self._load_analysis()
        self.conversion_stats = defaultdict(int)
        
    def _load_analysis(self):
        """Load the precision analysis results"""
        if not self.analysis_path.exists():
            raise FileNotFoundError(f"Analysis file not found: {self.analysis_path}")
        
        with open(self.analysis_path, 'r') as f:
            analysis = json.load(f)
        
        print(f"Loaded analysis for {analysis['total_tensors']} tensors")
        return analysis
    
    def get_tensor_precision(self, tensor_name):
        """Get recommended precision for a specific tensor"""
        if tensor_name in self.analysis['tensor_details']:
            details = self.analysis['tensor_details'][tensor_name]
            recommended = details['recommendation']['precision']
            
            # If analysis recommends FP32, keep it
            if recommended == 'fp32':
                return 'fp32'
            # If analysis recommends BF16 or FP16, use target precision
            elif recommended == 'bf16':
                return 'bf16' if self.target_precision == 'bf16' else 'fp16'
            elif recommended == 'fp16':
                return 'fp16'
        
        # Default to target precision if not found in analysis
        return self.target_precision
    
    def convert_tensor(self, tensor, target_dtype):
        """Convert a tensor to target dtype"""
        if target_dtype == 'fp32':
            return tensor.float()
        elif target_dtype == 'bf16':
            return tensor.bfloat16()
        elif target_dtype == 'fp16':
            return tensor.half()
        else:
            raise ValueError(f"Unknown dtype: {target_dtype}")
    
    def find_shard_files(self):
        """Find all shard files in the model directory"""
        safetensors_files = list(self.model_path.glob("*.safetensors"))
        bin_files = list(self.model_path.glob("*.bin"))
        
        if safetensors_files:
            print(f"Found {len(safetensors_files)} safetensors files")
            return safetensors_files, "safetensors"
        elif bin_files:
            print(f"Found {len(bin_files)} .bin files")
            return bin_files, "bin"
        else:
            raise FileNotFoundError(f"No model files found in {self.model_path}")
    
    def convert_shard(self, shard_path, output_path, file_format):
        """Convert a single shard to mixed precision"""
        print(f"\nConverting: {shard_path.name}")
        converted_state_dict = {}
        
        if file_format == "safetensors":
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                keys = f.keys()
                for key in tqdm(keys, desc="Converting tensors"):
                    tensor = f.get_tensor(key)
                    target_dtype = self.get_tensor_precision(key)
                    
                    # Convert tensor
                    converted_tensor = self.convert_tensor(tensor, target_dtype)
                    converted_state_dict[key] = converted_tensor
                    
                    # Track conversion stats
                    self.conversion_stats[target_dtype] += 1
                    
                    # Log critical conversions
                    if target_dtype == 'fp32':
                        print(f"  Keeping in FP32: {key}")
        else:  # .bin format
            state_dict = torch.load(shard_path, map_location="cpu")
            for key, tensor in tqdm(state_dict.items(), desc="Converting tensors"):
                target_dtype = self.get_tensor_precision(key)
                
                # Convert tensor
                converted_tensor = self.convert_tensor(tensor, target_dtype)
                converted_state_dict[key] = converted_tensor
                
                # Track conversion stats
                self.conversion_stats[target_dtype] += 1
                
                # Log critical conversions
                if target_dtype == 'fp32':
                    print(f"  Keeping in FP32: {key}")
        
        # Save converted shard
        if file_format == "safetensors":
            save_file(converted_state_dict, output_path)
        else:
            torch.save(converted_state_dict, output_path)
        
        print(f"✓ Saved: {output_path.name}")
        return len(converted_state_dict)
    
    def convert_all(self, output_dir):
        """Convert all shards in the model"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nOutput directory: {output_path}")
        
        # Find and convert all shards
        shard_files, file_format = self.find_shard_files()
        total_tensors = 0
        
        for shard_file in shard_files:
            output_file = output_path / shard_file.name
            num_tensors = self.convert_shard(shard_file, output_file, file_format)
            total_tensors += num_tensors
        
        # Copy configuration files
        self._copy_config_files(output_path)
        
        # Generate conversion report
        self.generate_report(output_path, total_tensors)
        
        return total_tensors
    
    def _copy_config_files(self, output_path):
        """Copy configuration and metadata files to output directory"""
        config_files = [
            'config.json',
            'model_index.json',
            'README.md',
            'model.safetensors.index.json',
            'pytorch_model.bin.index.json'
        ]
        
        print("\nCopying configuration files...")
        for config_file in config_files:
            src = self.model_path / config_file
            if src.exists():
                dst = output_path / config_file
                shutil.copy2(src, dst)
                print(f"  ✓ Copied: {config_file}")
    
    def generate_report(self, output_path, total_tensors):
        """Generate and save conversion report"""
        report = {
            'source_model': str(self.model_path),
            'output_model': str(output_path),
            'target_precision': self.target_precision,
            'total_tensors': total_tensors,
            'conversion_stats': dict(self.conversion_stats),
            'analysis_source': str(self.analysis_path),
        }
        
        # Calculate percentages
        report['percentages'] = {}
        for dtype, count in self.conversion_stats.items():
            percentage = (count / total_tensors * 100) if total_tensors > 0 else 0
            report['percentages'][dtype] = round(percentage, 2)
        
        # Save report
        report_path = output_path / 'conversion_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        self.print_summary(report)
        
        print(f"\n✓ Conversion report saved to: {report_path}")
    
    def print_summary(self, report):
        """Print conversion summary"""
        print("\n" + "="*80)
        print("MIXED PRECISION CONVERSION SUMMARY")
        print("="*80)
        
        print(f"\nSource: {report['source_model']}")
        print(f"Output: {report['output_model']}")
        print(f"Target precision: {report['target_precision'].upper()}")
        print(f"Total tensors converted: {report['total_tensors']}")
        
        print("\n" + "-"*80)
        print("CONVERSION BREAKDOWN")
        print("-"*80)
        
        for dtype in ['fp32', 'bf16', 'fp16']:
            count = report['conversion_stats'].get(dtype, 0)
            percentage = report['percentages'].get(dtype, 0)
            if count > 0:
                print(f"{dtype.upper()}: {count} tensors ({percentage}%)")
        
        print("\n" + "-"*80)
        print("MEMORY IMPACT")
        print("-"*80)
        
        # Estimate memory savings
        fp32_count = report['conversion_stats'].get('fp32', 0)
        reduced_count = total_tensors - fp32_count
        
        # Rough estimate: BF16/FP16 uses ~50% of FP32 memory
        estimated_savings = (reduced_count / report['total_tensors'] * 50) if report['total_tensors'] > 0 else 0
        
        print(f"\nEstimated memory reduction: ~{estimated_savings:.1f}%")
        print(f"({reduced_count} tensors converted to lower precision)")
        
        print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Convert diffusers model to mixed precision based on analysis"
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the source model directory"
    )
    parser.add_argument(
        "analysis_json",
        type=str,
        help="Path to the precision analysis JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for converted model"
    )
    parser.add_argument(
        "--target-precision",
        type=str,
        choices=['bf16', 'fp16'],
        default='bf16',
        help="Target precision for non-critical weights (default: bf16)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be converted without actually converting"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("MIXED PRECISION MODEL CONVERTER")
    print("="*80)
    print(f"\nSource model: {args.model_path}")
    print(f"Analysis file: {args.analysis_json}")
    print(f"Output directory: {args.output}")
    print(f"Target precision: {args.target_precision.upper()}")
    
    if args.dry_run:
        print("\n⚠️  DRY RUN MODE - No files will be modified")
    
    # Create converter
    converter = MixedPrecisionConverter(
        args.model_path,
        args.analysis_json,
        args.target_precision
    )
    
    if args.dry_run:
        print("\nDry run complete. Use without --dry-run to perform conversion.")
        return
    
    # Perform conversion
    print("\nStarting conversion...")
    total_tensors = converter.convert_all(args.output)
    
    print(f"\n✓ Conversion complete! {total_tensors} tensors processed.")
    print(f"✓ Mixed precision model saved to: {args.output}")


if __name__ == "__main__":
    main()