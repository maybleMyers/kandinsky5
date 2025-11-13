# Block Swapping for 20B Kandinsky I2V Pro Model

This document explains how to run the 20B parameter Kandinsky I2V Pro model on GPUs with limited VRAM using block swapping.

## Overview

The 20B Kandinsky I2V Pro model has significantly more parameters than the 2B Lite model:
- **Transformer blocks**: 60 (vs 32 in 2B model)
- **Model dimension**: 4096 (vs 1792 in 2B model)
- **FF dimension**: 16384 (vs 7168 in 2B model)

Block swapping allows running this large model on GPUs with 48GB or less VRAM by keeping only a subset of transformer blocks in GPU memory at once, dynamically swapping them in/out from CPU memory as needed.

## Architecture Comparison

| Component | 2B Lite Model | 20B Pro Model |
|-----------|--------------|---------------|
| Model Dimension | 1792 | 4096 |
| FF Dimension | 7168 | 16384 |
| Text Blocks | 2 | 4 |
| Visual Blocks | 32 | 60 |
| Time Dimension | 512 | 1024 |
| Axes Dimensions | [16, 24, 24] | [32, 48, 48] |
| Attention Type | Flash | NABLA |

## How Block Swapping Works

1. **Initialization**: Core model components (embeddings, RoPE, output layer, text transformer) are loaded into GPU memory
2. **Visual Blocks**: Only `blocks_in_memory` visual transformer blocks (e.g., 6 out of 60) are kept in GPU memory
3. **During Forward Pass**: As the model processes each layer:
   - The next block is prefetched to GPU
   - The current block is processed
   - Old blocks are offloaded to CPU when memory limit is reached
4. **Sequential Processing**: Blocks are processed one at a time, minimizing peak GPU memory usage

## Configuration

### Config File

Create or modify `configs/config_5s_i2v_pro_20b.yaml`:

```yaml
model:
  checkpoint_path: "./weights/model/kandinsky5_i2v_pro_20b_5s.safetensors"
  dit_params:
    model_dim: 4096
    ff_dim: 16384
    num_text_blocks: 4
    num_visual_blocks: 60
    # ... other 20B params

block_swap:
  enabled: true
  blocks_in_memory: 6  # Adjust based on your VRAM
```

### VRAM Requirements

| VRAM | Recommended Settings | Expected Performance |
|------|---------------------|---------------------|
| 80GB+ | `blocks_in_memory: 12-20` | Fastest, minimal swapping |
| 48GB | `blocks_in_memory: 6-8` | Good balance |
| 40GB | `blocks_in_memory: 4-5`, `offload: True` | Moderate speed |
| 24GB | `blocks_in_memory: 2-3`, `offload: True`, `quantized_qwen: True` | Slower, frequent swapping |

## Usage

### Python API

```python
import torch
from kandinsky import get_I2V_pipeline_with_block_swap

device_map = {
    "dit": torch.device('cuda:0'),
    "vae": torch.device('cuda:0'),
    "text_embedder": torch.device('cuda:0')
}

# Load pipeline with block swapping
pipe = get_I2V_pipeline_with_block_swap(
    device_map=device_map,
    conf_path="./configs/config_5s_i2v_pro_20b.yaml",
    blocks_in_memory=6,      # Override config if needed
    quantized_qwen=True,     # Use 4-bit Qwen to save ~12GB
    offload=False,           # Set True for extra memory savings
)

# Generate video
result = pipe(
    text="The Dragon breaths fire.",
    image="./assets/test_image.jpg",
    time_length=5,
    seed=137,
    num_steps=50,
    save_path='./output.mp4',
)
```

### Command Line

Run the example script:

```bash
python inference_example_i2v_20b.py
```

Or use the main test script:

```bash
python test.py \
  --config ./configs/config_5s_i2v_pro_20b.yaml \
  --prompt "The Dragon breaths fire." \
  --image "./assets/test_image.jpg" \
  --video_duration 5 \
  --qwen_quantization
```

## Memory Optimization Strategies

### 1. Adjust Blocks in Memory

The most impactful setting. Lower values = less memory but slower:

```python
# High memory (48GB+)
blocks_in_memory=6

# Medium memory (40GB)
blocks_in_memory=4

# Low memory (24GB)
blocks_in_memory=2
```

### 2. Enable Model Offloading

Offload VAE and text encoder to CPU when not in use:

```python
pipe = get_I2V_pipeline_with_block_swap(
    ...
    offload=True,
)
```

### 3. Quantize Qwen Encoder

Use 4-bit NF4 quantization (saves ~12GB):

```python
pipe = get_I2V_pipeline_with_block_swap(
    ...
    quantized_qwen=True,
)
```

### 4. Choose Attention Engine

Different engines have different memory profiles:

```python
# Memory efficient
attention_engine="sdpa"

# Fastest (if available)
attention_engine="flash_attention_3"

# Auto-select best available
attention_engine="auto"
```

## Performance Characteristics

### Speed vs Memory Trade-off

| blocks_in_memory | Relative Speed | Peak VRAM | Swapping Overhead |
|-----------------|----------------|-----------|-------------------|
| 12 | 100% (baseline) | ~45GB | Minimal |
| 6 | ~85% | ~38GB | Low |
| 4 | ~70% | ~34GB | Moderate |
| 2 | ~50% | ~28GB | High |

*Approximate values, actual performance varies by GPU and system*

### Bottlenecks

1. **PCIe Bandwidth**: Swapping blocks between CPU and GPU is limited by PCIe speed
2. **Block Size**: Each visual transformer block in the 20B model is ~650MB
3. **Synchronization**: GPU may idle while waiting for next block to transfer

### Optimization Tips

1. **PCIe Gen4 or higher**: Significantly faster transfers than Gen3
2. **High-speed CPU RAM**: Faster system memory helps
3. **Batch Size**: Keep batch size at 1 to minimize memory usage
4. **Video Duration**: Start with 5s videos before attempting longer generations

## Debugging

### Enable Debug Output

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Common Issues

**"Out of memory" errors**:
- Reduce `blocks_in_memory`
- Enable `offload=True`
- Enable `quantized_qwen=True`
- Reduce video resolution

**Slow generation**:
- Increase `blocks_in_memory` if you have VRAM headroom
- Check PCIe link speed: `nvidia-smi -q | grep -i pcie`
- Verify Flash Attention 3 is installed for optimal performance

**Incorrect output**:
- Verify model weights are loaded correctly
- Check NABLA attention parameters in config
- Try disabling block swap to isolate issue: `enable_block_swap=False`

## Implementation Details

### DiffusionTransformer3DBlockSwap

The block swapping implementation extends the base `DiffusionTransformer3D` class:

1. **Block Tracking**: Maintains `_blocks_on_gpu` set to track which blocks are in GPU memory
2. **FIFO Strategy**: When at capacity, offloads the oldest block
3. **Prefetching**: Loads the next block while processing the current one
4. **Non-blocking Transfers**: Uses `non_blocking=True` for async CPU-GPU transfers

### Key Methods

- `_ensure_block_on_gpu(block_idx, device)`: Loads a specific block to GPU
- `_prefetch_blocks(start_idx, device, num_blocks)`: Prefetches multiple blocks
- `forward()`: Modified to swap blocks during visual transformer processing

## Limitations

1. **Multi-GPU**: Block swapping with tensor parallelism is not currently supported
2. **MagCache**: Not tested with block swapping
3. **Compilation**: torch.compile may not work optimally with dynamic block swapping

## Future Improvements

Potential enhancements:
1. **Smart Prefetching**: Predict which blocks will be needed next
2. **Compression**: Compress blocks in CPU memory
3. **Double Buffering**: Overlap computation and transfer
4. **Adaptive Swapping**: Automatically adjust `blocks_in_memory` based on available VRAM

## Citation

If you use block swapping for the 20B model in your research, please cite the Kandinsky 5.0 paper and acknowledge this implementation.

## Support

For issues or questions:
1. Check the main README.md
2. Open an issue on GitHub
3. Ensure you're using the latest version of the code
