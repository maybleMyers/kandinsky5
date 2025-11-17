# INT8 Model Conversion Guide

This guide explains how to convert Kandinsky model checkpoints to INT8 format for reduced memory usage and faster inference.

## Benefits of INT8 Quantization

- **Memory Reduction**: ~50% reduction for quantized layers (75% vs FP32, 50% vs BF16)
- **Faster Inference**: 1.5-2x speedup with Triton INT8 kernels
- **Quality Preservation**: Minimal quality loss with block-wise quantization
- **Compatible**: Works with block swapping, NABLA attention, and MagCache

## Two Options for INT8 Usage

### Option 1: On-the-Fly Conversion (Easy, but requires GPU memory)

Simply add `--use_int8` flag when running inference:

```bash
python test.py \
  --config ./configs/config_5s_t2v_pro_20b.yaml \
  --use_int8 \
  --dtype bfloat16 \
  --prompt "Your prompt here"
```

**Note**: This requires enough GPU memory to load and quantize weights during loading. For large models like Pro 20B, this may not fit on a single GPU.

### Option 2: Pre-Conversion (Recommended for large models)

Pre-convert the checkpoint to INT8 format offline:

```bash
python scripts/convert_to_int8.py \
  --input_checkpoint Kandinsky-5.0-T2V-Pro-sft-5s-Diffusers/transformer/diffusion_pytorch_model.safetensors \
  --output_checkpoint Kandinsky-5.0-T2V-Pro-sft-5s-Diffusers/transformer/diffusion_pytorch_model_int8.safetensors \
  --block_size 128
```

Then load the pre-converted checkpoint:

```bash
python test.py \
  --config ./configs/config_5s_t2v_pro_20b.yaml \
  --use_int8 \
  --checkpoint_path_override Kandinsky-5.0-T2V-Pro-sft-5s-Diffusers/transformer/diffusion_pytorch_model_int8.safetensors \
  --dtype bfloat16 \
  --prompt "Your prompt here"
```

## Conversion Script Options

### Basic Usage

```bash
python scripts/convert_to_int8.py \
  --input_checkpoint <path/to/original.safetensors> \
  --output_checkpoint <path/to/output_int8.safetensors>
```

### All Options

- `--input_checkpoint`: Path to original checkpoint (safetensors format)
- `--output_checkpoint`: Path to save INT8 checkpoint
- `--block_size`: Block size for quantization (default: 128, **must be 128 for Triton kernels**)
- `--quiet`: Suppress progress output

### Example: Converting T2V Pro 20B Model

```bash
# Convert the T2V Pro 20B checkpoint
python scripts/convert_to_int8.py \
  --input_checkpoint Kandinsky-5.0-T2V-Pro-sft-5s-Diffusers/transformer/diffusion_pytorch_model.safetensors \
  --output_checkpoint Kandinsky-5.0-T2V-Pro-sft-5s-Diffusers/transformer/diffusion_pytorch_model_int8.safetensors
```

This will:
1. Load the checkpoint from disk
2. Process each linear layer one at a time
3. Quantize weights to INT8 with block-wise scaling
4. Save the INT8 checkpoint
5. Report memory savings

**Expected output:**
```
Loading checkpoint from Kandinsky-5.0-T2V-Pro-sft-5s-Diffusers/transformer/diffusion_pytorch_model.safetensors
Found 432 layers to convert to INT8
Keeping 124 layers in original format
Converting weights: 100%|██████████| 556/556 [05:23<00:00,  1.72it/s]

Conversion complete!
Original converted layers size: 18432.00 MB
INT8 converted layers size: 9216.00 MB
Size reduction: 50.0%

Saving INT8 checkpoint to Kandinsky-5.0-T2V-Pro-sft-5s-Diffusers/transformer/diffusion_pytorch_model_int8.safetensors
Done!
```

## Which Layers Are Quantized?

The conversion script automatically quantizes:

### ✅ Quantized Layers
- Attention query/key/value projections
- Attention output projections
- Feedforward input/output layers
- Cross-attention projections

### ❌ Kept in Original Precision
- Layer normalization
- Time embeddings
- Position embeddings
- Output projections
- Small layers (<1024 features)

## Using Pre-Quantized Checkpoints

After conversion, you can use the INT8 checkpoint with any pipeline:

### T2V with Block Swapping
```bash
python test.py \
  --config ./configs/config_5s_t2v_pro_20b.yaml \
  --use_int8 \
  --checkpoint_path_override path/to/model_int8.safetensors \
  --enable_block_swap \
  --blocks_in_memory 21 \
  --dtype bfloat16 \
  --prompt "Your prompt here"
```

### I2V
```bash
python test.py \
  --config ./configs/config_i2v.yaml \
  --use_int8 \
  --checkpoint_path_override path/to/model_int8.safetensors \
  --dtype bfloat16 \
  --image path/to/image.jpg \
  --prompt "Your prompt here"
```

## Memory Usage Comparison

For **Kandinsky Pro 20B model**:

| Configuration | GPU Memory | Notes |
|--------------|------------|-------|
| BF16 baseline | ~40 GB | Full model in BF16 |
| BF16 + Block Swap (21 blocks) | ~24 GB | Keep 21/48 blocks in VRAM |
| **INT8 + Block Swap (21 blocks)** | **~16 GB** | 50% reduction on quantized layers |
| INT8 + Block Swap (42 blocks) | ~28 GB | More blocks in memory |

## Verification

After conversion, verify the checkpoint loads correctly:

```bash
# This should print "loading pre-quantized INT8 checkpoint"
python test.py \
  --config ./configs/config_5s_t2v_pro_20b.yaml \
  --use_int8 \
  --checkpoint_path_override path/to/model_int8.safetensors \
  --prompt "test" \
  --sample_steps 1
```

## Troubleshooting

### Error: "in_features not divisible by block_size"

Some layers may have dimensions that aren't divisible by 128. These layers are automatically skipped and kept in original precision.

### Error: "CUDA out of memory" during conversion

The conversion script processes one layer at a time and should fit on most GPUs. If you still get OOM:
- Close other GPU processes
- Try on a machine with more VRAM
- The script is designed to minimize memory usage

### Quality Issues

If you notice quality degradation:
- INT8 quantization has minimal impact, but some prompts may be sensitive
- Try without INT8 to compare
- Block-wise quantization (block_size=128) provides better quality than per-tensor quantization

## Technical Details

The conversion uses:
- **Block-wise quantization**: Weights are quantized in 128-element blocks
- **Per-block scaling**: Each block has its own scale factor for better precision
- **INT8 storage**: Weights stored as int8, scales as float32
- **Triton kernels**: Custom CUDA kernels for efficient INT8 matmul

For more details, see `RamTorch/kintegration/README_TRITON_INT8.md`.
