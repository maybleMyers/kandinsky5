#!/usr/bin/env python3
"""
Enhanced Sharded Diffusers Model Precision Analyzer

Analyzes all weights in a sharded diffusers model to determine safe conversion
to fp8/int8/bf16/fp16, identifying weights that must remain in fp32.
Includes int8 and fp8 quantization analysis with advanced scaling and metrics.
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
import warnings
warnings.filterwarnings('ignore')


class EnhancedPrecisionAnalyzer:
    ### CHANGE ###
    # Added fp8_scaling_method to the constructor
    def __init__(self, model_path, config_path=None, fp8_scaling_method='per-channel'):
        self.model_path = Path(model_path)
        self.config = self._load_config(config_path)
        self.results = defaultdict(dict)
        self.int8_config = {
            'outlier_threshold': 6.0,
            'max_quantization_error': 0.05,
            'symmetric_quantization': True,
            'per_channel': True,
        }
        self.fp8_config = {
            'max_relative_error': 0.05, # Loosened default to 5% as a starting point
            'min_sqnr': 20.0, # Minimum acceptable Signal-to-Quantization-Noise Ratio (in dB)
            'scaling_method': fp8_scaling_method, # 'per-tensor' or 'per-channel'
            'block_size': 128, # For potential block-wise scaling in the future
        }
        self.fp8_supported = hasattr(torch, 'float8_e4m3fn')

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

    ### CHANGE ###
    # This function is completely rewritten for better analysis.
    def analyze_fp8_quantization(self, tensor):
        """Analyze tensor for FP8 quantization viability with advanced scaling."""
        fp8_stats = {}

        if not self.fp8_supported:
            fp8_stats['viable'] = False
            fp8_stats['reason'] = 'PyTorch version does not support float8 types.'
            return fp8_stats

        tensor_f32 = tensor.float()

        # We will focus on E5M2 as it's generally better for weights due to wider range
        fp8_dtype = torch.float8_e5m2
        fp8_max_val = torch.finfo(fp8_dtype).max

        if self.fp8_config['scaling_method'] == 'per-channel' and tensor_f32.ndim >= 2:
            # Per-channel along the output dimension (dim=0)
            orig_shape = tensor_f32.shape
            tensor_2d = tensor_f32.view(orig_shape[0], -1)
            
            # Calculate scale per channel
            scales = fp8_max_val / torch.max(torch.abs(tensor_2d), dim=1, keepdim=True).values.clamp(min=1e-12)
            
            # Quantize and dequantize
            quant = (tensor_2d * scales).to(fp8_dtype)
            dequant = (quant.to(torch.float32) / scales).view(orig_shape)
            fp8_stats['scaling_method'] = 'per-channel'

        else: # Default to per-tensor
            scale = fp8_max_val / torch.max(torch.abs(tensor_f32)).clamp(min=1e-12)
            quant = (tensor_f32 * scale).to(fp8_dtype)
            dequant = quant.to(torch.float32) / scale
            fp8_stats['scaling_method'] = 'per-tensor'

        # Calculate more robust error metrics
        error = tensor_f32 - dequant
        relative_error = (error.abs() / (tensor_f32.abs() + 1e-9)).mean()

        # Calculate Signal-to-Quantization-Noise Ratio (SQNR)
        signal_power = torch.sum(tensor_f32 ** 2)
        noise_power = torch.sum(error ** 2)
        sqnr = 10 * torch.log10(signal_power / noise_power.clamp(min=1e-12)) if noise_power > 0 else float('inf')

        fp8_stats['e5m2_relative_error'] = float(relative_error)
        fp8_stats['e5m2_sqnr_db'] = float(sqnr)

        # Viability Check using new metrics
        if sqnr >= self.fp8_config['min_sqnr']:
            fp8_stats['viable'] = True
            fp8_stats['reason'] = f'High SQNR ({sqnr:.2f} dB) with {fp8_stats["scaling_method"]} scaling.'
            fp8_stats['recommended_format'] = 'E5M2'
        elif relative_error < self.fp8_config['max_relative_error']:
            fp8_stats['viable'] = True
            fp8_stats['reason'] = f'Low relative error ({relative_error:.2%}) with {fp8_stats["scaling_method"]} scaling.'
            fp8_stats['recommended_format'] = 'E5M2'
        else:
            fp8_stats['viable'] = False
            fp8_stats['reason'] = f'Low SQNR ({sqnr:.2f} dB) and high error ({relative_error:.2%})'

        return fp8_stats

    def analyze_int8_quantization(self, tensor):
        """Analyze tensor for int8 quantization viability"""
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor)

        tensor_f32 = tensor.float() if tensor.dtype != torch.float32 else tensor
        int8_stats = {}

        if tensor.numel() < 1024:
            int8_stats['viable'] = False
            int8_stats['reason'] = 'Tensor too small for efficient quantization'
            return int8_stats

        abs_vals = tensor_f32.abs()
        mean_val = tensor_f32.mean()
        std_val = tensor_f32.std()

        outlier_threshold = mean_val.abs() + self.int8_config['outlier_threshold'] * std_val
        outliers = abs_vals > outlier_threshold
        num_outliers = outliers.sum().item()
        outlier_ratio = num_outliers / tensor.numel()

        int8_stats['outlier_ratio'] = float(outlier_ratio)
        int8_stats['num_outliers'] = int(num_outliers)

        if outlier_ratio > 0.01:
            int8_stats['viable'] = False
            int8_stats['reason'] = f'Too many outliers ({outlier_ratio:.2%})'
            return int8_stats

        if self.int8_config['per_channel'] and len(tensor.shape) >= 2:
            quant_results = self._simulate_per_channel_quantization(tensor_f32)
        else:
            quant_results = self._simulate_per_tensor_quantization(tensor_f32)

        int8_stats.update(quant_results)

        if quant_results['max_error'] > self.int8_config['max_quantization_error']:
            int8_stats['viable'] = False
            int8_stats['reason'] = f'Quantization error too high ({quant_results["max_error"]:.2%})'
        elif quant_results['mean_error'] > self.int8_config['max_quantization_error'] / 2:
            int8_stats['viable'] = False
            int8_stats['reason'] = f'Mean quantization error too high ({quant_results["mean_error"]:.2%})'
        else:
            int8_stats['viable'] = True
            int8_stats['reason'] = 'Suitable for int8 quantization'
            original_bits = 32 if tensor.dtype == torch.float32 else 16
            int8_stats['compression_ratio'] = original_bits / 8
            int8_stats['memory_reduction'] = f'{(1 - 8/original_bits) * 100:.1f}%'
        return int8_stats

    def _simulate_per_tensor_quantization(self, tensor):
        """Simulate per-tensor int8 quantization"""
        if self.int8_config['symmetric_quantization']:
            max_val = tensor.abs().max()
            scale = max_val / 127.0
            zero_point = 0
        else:
            min_val = tensor.min()
            max_val = tensor.max()
            scale = (max_val - min_val) / 255.0
            zero_point = -min_val / scale

        if scale > 0:
            quantized = torch.round(tensor / scale + zero_point)
            quantized = torch.clamp(quantized, -128 if self.int8_config['symmetric_quantization'] else 0,
                                   127 if self.int8_config['symmetric_quantization'] else 255)
            dequantized = (quantized - zero_point) * scale
            errors = (tensor - dequantized).abs()
            relative_errors = errors / (tensor.abs() + 1e-8)
            return {
                'quantization_type': 'per_tensor', 'scale': float(scale), 'zero_point': float(zero_point),
                'max_error': float(relative_errors.max()), 'mean_error': float(relative_errors.mean()),
                'rmse': float(torch.sqrt(torch.mean(errors ** 2))),
            }
        else:
            return {'quantization_type': 'per_tensor', 'scale': 0, 'zero_point': 0, 'max_error': 0, 'mean_error': 0, 'rmse': 0}

    def _simulate_per_channel_quantization(self, tensor):
        """Simulate per-channel int8 quantization"""
        orig_shape = tensor.shape
        if tensor.ndim < 2: return self._simulate_per_tensor_quantization(tensor) # Fallback for 1D
        tensor_2d = tensor.view(tensor.shape[0], -1)
        
        scales = []
        zero_points = []
        dequantized_channels = []
        
        for channel_idx in range(tensor_2d.shape[0]):
            channel = tensor_2d[channel_idx]
            if self.int8_config['symmetric_quantization']:
                max_val = channel.abs().max()
                scale = max_val / 127.0 if max_val > 0 else 1.0
                zero_point = 0
            else:
                min_val, max_val = channel.min(), channel.max()
                scale = (max_val - min_val) / 255.0 if max_val > min_val else 1.0
                zero_point = -min_val / scale
            scales.append(scale)
            zero_points.append(zero_point)
            
            if scale > 0:
                quantized = torch.round(channel / scale + zero_point)
                quantized = torch.clamp(quantized, -128 if self.int8_config['symmetric_quantization'] else 0,
                                       127 if self.int8_config['symmetric_quantization'] else 255)
                dequantized = (quantized - zero_point) * scale
                dequantized_channels.append(dequantized)
            else:
                dequantized_channels.append(channel) # Append original if scale is zero
        
        if dequantized_channels:
            dequantized_tensor_2d = torch.stack(dequantized_channels)
            dequantized_tensor = dequantized_tensor_2d.view(orig_shape)
            errors = (tensor - dequantized_tensor).abs()
            relative_errors = errors / (tensor.abs() + 1e-8)
            return {
                'quantization_type': 'per_channel', 'num_channels': len(scales),
                'scale_range': [float(min(s.item() for s in scales)), float(max(s.item() for s in scales))],
                'max_error': float(relative_errors.max()), 'mean_error': float(relative_errors.mean()),
                'rmse': float(torch.sqrt(torch.mean(errors ** 2))),
            }
        else:
            return {'quantization_type': 'per_channel', 'num_channels': 0, 'scale_range': [0, 0], 'max_error': 0, 'mean_error': 0, 'rmse': 0}

    def analyze_tensor(self, name, tensor):
        """Analyze a single tensor for precision conversion safety"""
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor)

        if tensor.dtype in [torch.bfloat16, torch.float16]:
            tensor_f32 = tensor.float()
        else:
            tensor_f32 = tensor

        stats = {
            'dtype': str(tensor.dtype), 'shape': list(tensor.shape), 'num_elements': tensor.numel(),
            'min': float(tensor_f32.min()), 'max': float(tensor_f32.max()),
            'mean': float(tensor_f32.mean()), 'std': float(tensor_f32.std()),
            'abs_max': float(tensor_f32.abs().max()),
        }

        stats['has_nan'] = bool(torch.isnan(tensor_f32).any())
        stats['has_inf'] = bool(torch.isinf(tensor_f32).any())
        stats['num_zeros'] = int((tensor_f32 == 0).sum())
        stats['sparsity'] = float(stats['num_zeros'] / stats['num_elements'])

        fp16_min, fp16_max = 6e-5, 65504
        abs_vals = tensor_f32.abs()
        stats['fp16_safe_range'] = bool(
            (abs_vals[abs_vals > 0].min() >= fp16_min) and
            (abs_vals.max() <= fp16_max)
        ) if abs_vals[abs_vals > 0].numel() > 0 else True
        stats['bf16_safe_range'] = True

        if tensor.dtype == torch.float32:
            tensor_fp16 = tensor_f32.half().float()
            fp16_error = (tensor_f32 - tensor_fp16).abs()
            stats['fp16_relative_error'] = float((fp16_error / (tensor_f32.abs() + 1e-8)).mean())
            tensor_bf16 = tensor_f32.bfloat16().float()
            bf16_error = (tensor_f32 - tensor_bf16).abs()
            stats['bf16_relative_error'] = float((bf16_error / (tensor_f32.abs() + 1e-8)).mean())

        stats['int8_analysis'] = self.analyze_int8_quantization(tensor_f32)
        stats['fp8_analysis'] = self.analyze_fp8_quantization(tensor_f32)
        stats['recommendation'] = self._make_recommendation(name, stats)
        return stats

    ### CHANGE ###
    # Updated recommendation logic to use the new FP8 stats
    def _make_recommendation(self, name, stats):
        """Make precision recommendation based on analysis including FP8"""
        critical_patterns = ['norm', 'ln', 'layernorm', 'groupnorm', 'embedding', 'position', 'time_embed', 'scale', 'shift', 'gamma', 'beta']
        quant_friendly_patterns = ['weight', 'kernel', 'conv', 'linear', 'projection', 'dense', 'fc']

        name_lower = name.lower()
        is_critical = any(pattern in name_lower for pattern in critical_patterns)
        is_quant_friendly = any(pattern in name_lower for pattern in quant_friendly_patterns)
        
        if stats['has_nan'] or stats['has_inf']:
            return {'precision': 'fp32', 'reason': 'Contains NaN or Inf values', 'confidence': 'high'}
        if is_critical:
            return {'precision': 'fp32', 'reason': f'Critical layer type: {name}', 'confidence': 'high'}

        # FP8 check is now first and more robust
        if stats['fp8_analysis'].get('viable', False) and is_quant_friendly and stats['num_elements'] > 4096:
             return {'precision': 'fp8', 'reason': stats['fp8_analysis']['reason'], 'confidence': 'high', 'recommended_format': stats['fp8_analysis']['recommended_format']}

        int8_viable = stats['int8_analysis']['viable']
        if int8_viable and is_quant_friendly and stats['num_elements'] > 4096:
            return {'precision': 'int8', 'reason': stats['int8_analysis']['reason'], 'confidence': 'high'}

        if 'bf16_relative_error' in stats and stats['bf16_relative_error'] < 0.005:
            return {'precision': 'bf16', 'reason': f'Low BF16 error: {stats["bf16_relative_error"]:.4f}', 'confidence': 'high'}
        
        if 'fp16_relative_error' in stats and stats['fp16_relative_error'] < 0.01:
            return {'precision': 'fp16', 'reason': f'Low FP16 error: {stats["fp16_relative_error"]:.4f}', 'confidence': 'medium'}

        if stats.get('abs_max', 0) > 0 and stats.get('min', 0) != 0:
            abs_min_nonzero = torch.min(torch.abs(torch.tensor(stats['abs_max']))[torch.abs(torch.tensor(stats['abs_max'])) > 0]) if stats.get('abs_max', 0) > 0 else 1e-10
            dynamic_range = stats['abs_max'] / (abs_min_nonzero + 1e-10)
            if dynamic_range > 1e6:
                return {'precision': 'fp32', 'reason': f'High dynamic range: {dynamic_range:.2e}', 'confidence': 'high'}

        if not stats['bf16_safe_range']:
             return {'precision': 'fp32', 'reason': 'Values outside BF16 range', 'confidence': 'high'}
        elif not stats['fp16_safe_range']:
             return {'precision': 'bf16', 'reason': 'Values outside FP16 range but within BF16', 'confidence': 'high'}

        return {'precision': 'bf16', 'reason': 'General weight tensor, safe for BF16', 'confidence': 'medium'}
        
    def analyze_shard(self, shard_path, file_format):
        print(f"\nAnalyzing: {shard_path.name}")
        if file_format == "safetensors":
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                keys = list(f.keys())
                for key in tqdm(keys, desc="Tensors"):
                    self.results[key] = self.analyze_tensor(key, f.get_tensor(key))
        else:
            state_dict = torch.load(shard_path, map_location="cpu")
            for key, tensor in tqdm(state_dict.items(), desc="Tensors"):
                self.results[key] = self.analyze_tensor(key, tensor)

    def analyze_all(self):
        shard_files, file_format = self.find_shard_files()
        for shard_file in shard_files:
            self.analyze_shard(shard_file, file_format)
        return self.generate_report()

    def generate_report(self):
        # ... (This function remains mostly the same, no changes needed)
        report = {
            'model_path': str(self.model_path), 'config': self.config, 'total_tensors': len(self.results),
            'summary': defaultdict(int), 'recommendations': defaultdict(list),
            'int8_statistics': {'total_int8_viable': 0, 'int8_memory_savings': 0, 'int8_tensors': []},
            'critical_tensors': [], 'detailed_results': {}
        }
        total_params, int8_params, fp8_params = 0, 0, 0
        
        for name, stats in self.results.items():
            rec, precision, num_params = stats['recommendation'], stats['recommendation']['precision'], stats['num_elements']
            total_params += num_params
            report['summary'][precision] += 1
            report['recommendations'][precision].append({'name': name, 'reason': rec['reason'], 'confidence': rec['confidence']})
            
            if precision == 'fp8':
                fp8_params += num_params
            
            if rec.get('int8_viable', False) or precision == 'int8':
                report['int8_statistics']['total_int8_viable'] += 1
                if precision == 'int8':
                    int8_params += num_params
                    report['int8_statistics']['int8_tensors'].append({'name': name, 'shape': stats['shape'], 'quantization_error': stats['int8_analysis'].get('mean_error', 0)})

            if precision == 'fp32' and rec['confidence'] == 'high':
                report['critical_tensors'].append({'name': name, 'reason': rec['reason'], 'shape': stats['shape'], 'abs_max': stats['abs_max']})
            
            # Storing detailed results can make the JSON huge, this part can be trimmed if needed
            report['detailed_results'][name] = stats
        
        if total_params > 0:
            # Add FP8 stats to the report summary
            report['fp8_statistics'] = {
                'total_fp8_viable': report['summary']['fp8'],
                'fp8_params_percentage': (fp8_params / total_params) * 100 if total_params > 0 else 0,
                'potential_memory_savings': ((fp8_params * 3) / (total_params * 4)) * 100 if total_params > 0 else 0
            }
            original_memory = total_params * 4 # Assuming original is fp32
            int8_saved_memory = (int8_params * 3)
            report['int8_statistics']['int8_memory_savings'] = (int8_saved_memory / original_memory) * 100 if original_memory > 0 else 0
            report['int8_statistics']['int8_params_percentage'] = (int8_params / total_params) * 100 if total_params > 0 else 0
        return report

    def print_report(self, report):
        print("\n" + "="*80 + "\nENHANCED MODEL PRECISION ANALYSIS REPORT\n" + "="*80)
        print(f"\nModel: {report['model_path']}\nTotal tensors analyzed: {report['total_tensors']}")
        if self.fp8_supported: print(f"FP8 analysis: Enabled (Scaling: {self.fp8_config['scaling_method']})")
        else: print("FP8 analysis: Disabled (PyTorch version too old for float8)")
        print("\n" + "-"*80 + "\nRECOMMENDATIONS SUMMARY\n" + "-"*80)

        for precision in ['fp8', 'int8', 'fp32', 'bf16', 'fp16']:
            if precision in report['summary']:
                count = report['summary'][precision]
                percentage = (count / report['total_tensors'] * 100) if report['total_tensors'] > 0 else 0
                print(f"{precision.upper()}: {count} tensors ({percentage:.1f}%)")

        if report['summary']['fp8'] > 0:
            print("\n" + "-"*80 + "\nFP8 QUANTIZATION ANALYSIS\n" + "-"*80)
            fp8_stats = report['fp8_statistics']
            print(f"\nTotal FP8 recommended tensors: {fp8_stats['total_fp8_viable']}")
            print(f"Parameters convertible to FP8: {fp8_stats['fp8_params_percentage']:.1f}%")
            print(f"Potential memory savings from FP8 alone on these params: ~75%")
            print("These tensors can be converted to scaled FP8 with high SQNR or low error.")
            print("Recommended format for weights is E5M2 for its superior dynamic range.")


        if report['summary']['int8'] > 0:
            print("\n" + "-"*80 + "\nINT8 QUANTIZATION ANALYSIS\n" + "-"*80)
            int8_stats = report['int8_statistics']
            print(f"\nTotal INT8 viable tensors: {int8_stats['total_int8_viable']}\nRecommended for INT8: {report['summary']['int8']} tensors")
            if int8_stats.get('int8_params_percentage', 0) > 0:
                print(f"Parameters convertible to INT8: {int8_stats['int8_params_percentage']:.1f}%")
                print(f"Potential memory savings from INT8: {int8_stats['int8_memory_savings']:.1f}% of total model size")

        print("\n" + "-"*80 + "\nCRITICAL TENSORS (MUST STAY FP32)\n" + "-"*80)
        if report['critical_tensors']:
            for item in report['critical_tensors'][:15]:
                print(f"\n• {item['name']}\n  Reason: {item['reason']}\n  Shape: {item['shape']}, Max value: {item['abs_max']:.6e}")
            if len(report['critical_tensors']) > 15: print(f"\n... and {len(report['critical_tensors']) - 15} more")
        else: print("\nNo critical tensors requiring FP32 found.")

        print("\n" + "="*80)

    def save_report(self, report, output_path):
        # ... (This function remains mostly the same, no changes needed)
        with open(output_path, 'w') as f:
            # A custom encoder to handle potential numpy types if they sneak in
            class NpEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer): return int(obj)
                    if isinstance(obj, np.floating): return float(obj)
                    if isinstance(obj, np.ndarray): return obj.tolist()
                    return super(NpEncoder, self).default(obj)
            json.dump(report, f, indent=2, cls=NpEncoder)
        print(f"\n✓ Detailed report saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Analyze sharded diffusers model for precision conversion including FP8/INT8 quantization"
    )
    parser.add_argument("model_path", type=str, help="Path to the model directory containing shard files")
    parser.add_argument("--config", type=str, default=None, help="Path to config.json (default: model_path/config.json)")
    parser.add_argument("--output", type=str, default="fp8_precision_analysis.json", help="Output path for detailed JSON report")
    ### CHANGE ###
    # Added new arguments for more control over the analysis
    parser.add_argument("--fp8-scaling", type=str, default="per-channel", choices=['per-tensor', 'per-channel'], help="Scaling method for FP8 analysis")
    parser.add_argument("--min-sqnr", type=float, default=20.0, help="Minimum acceptable SQNR in dB for FP8 viability (default: 20.0)")
    parser.add_argument("--max-fp8-error", type=float, default=0.05, help="Maximum acceptable relative error for FP8 if SQNR check fails (default: 0.05 = 5%%)")
    parser.add_argument("--outlier-threshold", type=float, default=6.0, help="Standard deviations for outlier detection in int8 analysis (default: 6.0)")
    parser.add_argument("--max-quant-error", type=float, default=0.05, help="Maximum acceptable quantization error for int8 (default: 0.05 = 5%%)")
    
    args = parser.parse_args()
    
    print("Starting enhanced model precision analysis...")
    print(f"Model path: {args.model_path}")
    
    ### CHANGE ###
    # Pass the new scaling method to the analyzer
    analyzer = EnhancedPrecisionAnalyzer(args.model_path, args.config, fp8_scaling_method=args.fp8_scaling)
    
    # Update configs from command-line arguments
    analyzer.fp8_config['max_relative_error'] = args.max_fp8_error
    analyzer.fp8_config['min_sqnr'] = args.min_sqnr
    analyzer.int8_config['outlier_threshold'] = args.outlier_threshold
    analyzer.int8_config['max_quantization_error'] = args.max_quant_error
    
    report = analyzer.analyze_all()
    
    analyzer.print_report(report)
    analyzer.save_report(report, args.output)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()