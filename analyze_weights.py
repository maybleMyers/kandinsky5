#!/usr/bin/env python3
"""
Sharded Diffusers Model Precision Analyzer

Analyzes all weights in a sharded diffusers model to determine safe conversion
to bf16/fp16, identifying weights that must remain in fp32.
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from safetensors import safe_open
from tqdm import tqdm


class PrecisionAnalyzer:
    def __init__(self, model_path, config_path=None):
        self.model_path = Path(model_path)
        self.config = self._load_config(config_path)
        self.results = defaultdict(dict)
        
    def _load_config(self, config_path):
        """Load model configuration"""
        if config_path:
            config_file = Path(config_path)
        else:
            config_file = self.model_path / "config.json"
            
        if config_file.exists():
            with open(config_file, 'r') as f:
                return json.load(f)
        return {}
    
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
    
    def analyze_tensor(self, name, tensor):
        """Analyze a single tensor for precision conversion safety"""
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor)
        
        # Convert to float32 for analysis
        if tensor.dtype in [torch.bfloat16, torch.float16]:
            tensor_f32 = tensor.float()
        else:
            tensor_f32 = tensor
        
        # Basic statistics
        stats = {
            'dtype': str(tensor.dtype),
            'shape': list(tensor.shape),
            'num_elements': tensor.numel(),
            'min': float(tensor_f32.min()),
            'max': float(tensor_f32.max()),
            'mean': float(tensor_f32.mean()),
            'std': float(tensor_f32.std()),
            'abs_max': float(tensor_f32.abs().max()),
        }
        
        # Check for special values
        stats['has_nan'] = bool(torch.isnan(tensor_f32).any())
        stats['has_inf'] = bool(torch.isinf(tensor_f32).any())
        stats['num_zeros'] = int((tensor_f32 == 0).sum())
        stats['sparsity'] = float(stats['num_zeros'] / stats['num_elements'])
        
        # Precision analysis
        abs_vals = tensor_f32.abs()
        
        # FP16 range: ~6e-8 to 65504
        # BF16 range: ~1e-45 to 3.4e38 (same exponent range as FP32)
        fp16_min, fp16_max = 6e-5, 65504  # Conservative limits
        bf16_min, bf16_max = 1e-38, 3.4e38
        
        # Check range compatibility
        stats['fp16_safe_range'] = bool(
            (abs_vals[abs_vals > 0].min() >= fp16_min) and 
            (abs_vals.max() <= fp16_max)
        ) if abs_vals[abs_vals > 0].numel() > 0 else True
        
        stats['bf16_safe_range'] = bool(
            (abs_vals[abs_vals > 0].min() >= bf16_min) and 
            (abs_vals.max() <= bf16_max)
        ) if abs_vals[abs_vals > 0].numel() > 0 else True
        
        # Test actual conversion quality
        if tensor.dtype == torch.float32:
            # Test FP16 conversion
            tensor_fp16 = tensor_f32.half().float()
            fp16_error = (tensor_f32 - tensor_fp16).abs()
            stats['fp16_max_error'] = float(fp16_error.max())
            stats['fp16_mean_error'] = float(fp16_error.mean())
            stats['fp16_relative_error'] = float(
                (fp16_error / (tensor_f32.abs() + 1e-8)).mean()
            )
            
            # Test BF16 conversion
            tensor_bf16 = tensor_f32.bfloat16().float()
            bf16_error = (tensor_f32 - tensor_bf16).abs()
            stats['bf16_max_error'] = float(bf16_error.max())
            stats['bf16_mean_error'] = float(bf16_error.mean())
            stats['bf16_relative_error'] = float(
                (bf16_error / (tensor_f32.abs() + 1e-8)).mean()
            )
        
        # Determine recommendations
        stats['recommendation'] = self._make_recommendation(name, stats)
        
        return stats
    
    def _make_recommendation(self, name, stats):
        """Make precision recommendation based on analysis"""
        # Critical patterns that should stay in FP32
        critical_patterns = [
            'norm', 'ln', 'layernorm', 'groupnorm', 
            'embedding', 'position', 'time_embed',
            'scale', 'shift', 'gamma', 'beta'
        ]
        
        name_lower = name.lower()
        
        # Check if name matches critical patterns
        is_critical = any(pattern in name_lower for pattern in critical_patterns)
        
        # Precision thresholds
        fp16_rel_error_threshold = 0.01  # 1% relative error
        bf16_rel_error_threshold = 0.005  # 0.5% relative error
        
        if stats['has_nan'] or stats['has_inf']:
            return {
                'precision': 'fp32',
                'reason': 'Contains NaN or Inf values',
                'confidence': 'high'
            }
        
        if is_critical:
            return {
                'precision': 'fp32',
                'reason': f'Critical layer type: {name}',
                'confidence': 'high'
            }
        
        # Check if very small values exist
        if stats['abs_max'] > 0 and stats['min'] != 0:
            dynamic_range = stats['abs_max'] / (abs(stats['min']) + 1e-10)
            if dynamic_range > 1e6:
                return {
                    'precision': 'fp32',
                    'reason': f'High dynamic range: {dynamic_range:.2e}',
                    'confidence': 'high'
                }
        
        # Check conversion errors if available
        if 'bf16_relative_error' in stats and 'fp16_relative_error' in stats:
            if stats['bf16_relative_error'] < bf16_rel_error_threshold:
                return {
                    'precision': 'bf16',
                    'reason': f'Low BF16 error: {stats["bf16_relative_error"]:.4f}',
                    'confidence': 'high'
                }
            elif stats['fp16_relative_error'] < fp16_rel_error_threshold:
                return {
                    'precision': 'fp16',
                    'reason': f'Low FP16 error: {stats["fp16_relative_error"]:.4f}',
                    'confidence': 'medium'
                }
            else:
                return {
                    'precision': 'fp32',
                    'reason': f'High conversion errors (BF16: {stats["bf16_relative_error"]:.4f}, FP16: {stats["fp16_relative_error"]:.4f})',
                    'confidence': 'high'
                }
        
        # Range-based fallback
        if not stats['bf16_safe_range']:
            return {
                'precision': 'fp32',
                'reason': 'Values outside BF16 range',
                'confidence': 'high'
            }
        elif not stats['fp16_safe_range']:
            return {
                'precision': 'bf16',
                'reason': 'Values outside FP16 range but within BF16',
                'confidence': 'high'
            }
        
        # Default to BF16 for general weights
        return {
            'precision': 'bf16',
            'reason': 'General weight tensor, safe for BF16',
            'confidence': 'medium'
        }
    
    def analyze_shard(self, shard_path, file_format):
        """Analyze all tensors in a single shard"""
        print(f"\nAnalyzing: {shard_path.name}")
        
        if file_format == "safetensors":
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                keys = f.keys()
                for key in tqdm(keys, desc="Tensors"):
                    tensor = f.get_tensor(key)
                    self.results[key] = self.analyze_tensor(key, tensor)
        else:  # .bin format
            state_dict = torch.load(shard_path, map_location="cpu")
            for key, tensor in tqdm(state_dict.items(), desc="Tensors"):
                self.results[key] = self.analyze_tensor(key, tensor)
    
    def analyze_all(self):
        """Analyze all shards in the model"""
        shard_files, file_format = self.find_shard_files()
        
        for shard_file in shard_files:
            self.analyze_shard(shard_file, file_format)
        
        return self.generate_report()
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        report = {
            'model_path': str(self.model_path),
            'config': self.config,
            'total_tensors': len(self.results),
            'summary': defaultdict(int),
            'recommendations': defaultdict(list),
            'critical_tensors': [],
            'detailed_results': {}
        }
        
        # Summarize recommendations
        for name, stats in self.results.items():
            rec = stats['recommendation']
            precision = rec['precision']
            report['summary'][precision] += 1
            report['recommendations'][precision].append({
                'name': name,
                'reason': rec['reason'],
                'confidence': rec['confidence']
            })
            
            # Track critical tensors (must stay FP32)
            if precision == 'fp32' and rec['confidence'] == 'high':
                report['critical_tensors'].append({
                    'name': name,
                    'reason': rec['reason'],
                    'shape': stats['shape'],
                    'abs_max': stats['abs_max']
                })
            
            # Store detailed results
            report['detailed_results'][name] = stats
        
        return report
    
    def print_report(self, report):
        """Print formatted analysis report"""
        print("\n" + "="*80)
        print("MODEL PRECISION ANALYSIS REPORT")
        print("="*80)
        
        print(f"\nModel: {report['model_path']}")
        print(f"Total tensors analyzed: {report['total_tensors']}")
        
        if report['config']:
            print(f"\nModel class: {report['config'].get('_class_name', 'Unknown')}")
        
        print("\n" + "-"*80)
        print("RECOMMENDATIONS SUMMARY")
        print("-"*80)
        
        for precision in ['fp32', 'bf16', 'fp16']:
            count = report['summary'][precision]
            percentage = (count / report['total_tensors'] * 100) if report['total_tensors'] > 0 else 0
            print(f"{precision.upper()}: {count} tensors ({percentage:.1f}%)")
        
        print("\n" + "-"*80)
        print("CRITICAL TENSORS (MUST STAY FP32)")
        print("-"*80)
        
        if report['critical_tensors']:
            for item in report['critical_tensors'][:20]:  # Show first 20
                print(f"\n• {item['name']}")
                print(f"  Reason: {item['reason']}")
                print(f"  Shape: {item['shape']}, Max value: {item['abs_max']:.6e}")
            
            if len(report['critical_tensors']) > 20:
                print(f"\n... and {len(report['critical_tensors']) - 20} more")
        else:
            print("\nNo critical tensors requiring FP32 found.")
        
        print("\n" + "-"*80)
        print("CONVERSION STRATEGY")
        print("-"*80)
        
        if report['summary']['bf16'] > report['summary']['fp16']:
            print("\n✓ Recommended: Convert to BF16")
            print("  BF16 provides better range and stability for this model.")
        elif report['summary']['fp16'] > 0:
            print("\n✓ Recommended: Convert to FP16 (with exceptions)")
            print("  FP16 is viable for most weights, but some require FP32.")
        
        if report['summary']['fp32'] > 0:
            print(f"\n⚠ Keep {report['summary']['fp32']} tensors in FP32")
            print("  These tensors have high precision requirements.")
        
        print("\n" + "="*80)
    
    def save_report(self, report, output_path):
        """Save detailed report to JSON"""
        # Remove torch tensors for JSON serialization
        clean_report = {
            'model_path': report['model_path'],
            'config': report['config'],
            'total_tensors': report['total_tensors'],
            'summary': dict(report['summary']),
            'recommendations': {k: v for k, v in report['recommendations'].items()},
            'critical_tensors': report['critical_tensors'],
        }
        
        # Add simplified detailed results
        clean_report['tensor_details'] = {}
        for name, stats in report['detailed_results'].items():
            clean_report['tensor_details'][name] = {
                'dtype': stats['dtype'],
                'shape': stats['shape'],
                'recommendation': stats['recommendation'],
                'abs_max': stats['abs_max'],
                'sparsity': stats['sparsity']
            }
        
        with open(output_path, 'w') as f:
            json.dump(clean_report, f, indent=2)
        
        print(f"\n✓ Detailed report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze sharded diffusers model for precision conversion safety"
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the model directory containing shard files"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.json (default: model_path/config.json)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="precision_analysis.json",
        help="Output path for detailed JSON report"
    )
    
    args = parser.parse_args()
    
    print("Starting model precision analysis...")
    print(f"Model path: {args.model_path}")
    
    analyzer = PrecisionAnalyzer(args.model_path, args.config)
    report = analyzer.analyze_all()
    
    analyzer.print_report(report)
    analyzer.save_report(report, args.output)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()