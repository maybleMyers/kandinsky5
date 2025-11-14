# Kandinsky 5 GUI

A simple Gradio-based GUI for generating videos with Kandinsky 5 models.

## Features

- **Text-to-Video (T2V)**: Generate videos from text prompts
- **Image-to-Video (I2V)**: Animate images based on text descriptions
- **Multiple Models**: Support for all Kandinsky 5 configurations (5s, 10s, I2V, 20B)
- **Memory Optimization**: Options for model offloading, Qwen quantization, and block swapping
- **Progress Tracking**: Real-time generation progress monitoring
- **Easy-to-Use Interface**: Clean, intuitive UI inspired by H1111

## Installation

1. Install the base Kandinsky 5 dependencies (see main README)

2. Install Gradio:
```bash
pip install gradio
```

## Usage

### Starting the GUI

Simply run:
```bash
python kandinsky_gui.py
```

The GUI will be available at `http://localhost:7860`

### Basic Workflow

1. **Select a Model**: Choose from the dropdown (e.g., "5s SFT (Recommended)" for T2V or "5s I2V" for I2V)

2. **Configure Memory Settings** (Optional):
   - Enable **Model Offloading** to save VRAM
   - Use **Quantized Qwen** (recommended, saves ~12GB VRAM)
   - For the 20B model, enable **Block Swapping** with 4-6 blocks in memory

3. **Load the Model**: Click "Load/Reload Model" button

4. **Generate Videos**:
   - **T2V Tab**: Enter your prompt and adjust parameters
   - **I2V Tab**: Upload an image, enter a prompt describing the desired motion

5. **Click Generate**: Your video will be saved to the `./outputs/` directory

## Parameters Guide

### Generation Parameters

- **Prompt**: Describe the video content or motion
- **Negative Prompt**: Describe what to avoid in the generation
- **Width/Height**: Video resolution (512x512, 512x768, or 768x512)
- **Duration**: Video length in seconds (1-10s, depending on model)
- **Steps**: Number of diffusion steps (20-100, default: 50)
- **Guidance Weight**: Classifier-free guidance strength (0-20, default: 7.5)
- **Scheduler Scale**: Noise scheduler scaling factor (1.0-10.0, default: 5.0)
- **Seed**: Random seed for reproducibility
- **Expand Prompt**: Use AI to expand and enhance your prompt

### Memory Optimization

- **Model Offloading**: Moves models to CPU when not in use (slower but saves VRAM)
- **Quantized Qwen**: Uses 4-bit quantization for the text embedder (saves ~12GB)
- **Block Swapping**: For 20B model - keeps only N transformer blocks in GPU memory
  - 6 blocks: ~48GB VRAM
  - 4 blocks: ~40GB VRAM
  - 2 blocks: ~32GB VRAM (slower)

### Attention Engines

- **auto**: Automatically selects the best available engine
- **flash_attention_3**: Fastest (if available)
- **flash_attention_2**: Fast and stable
- **sdpa**: PyTorch scaled dot-product attention
- **sage**: Alternative attention implementation

## Tips

1. **First Time**: Start with "5s SFT (Recommended)" model and default settings
2. **Low VRAM**: Enable all optimization options (offloading, quantized Qwen)
3. **20B Model**: Requires enable_block_swap=True with 4-6 blocks in memory
4. **Best Quality**: Use 50 steps, guidance weight 7.5, and prompt expansion
5. **Speed**: Reduce steps to 30-40 for faster generation (slight quality trade-off)

## Troubleshooting

### Out of Memory Errors

If you encounter CUDA OOM errors:
1. Enable model offloading
2. Ensure quantized Qwen is enabled
3. For 20B model, reduce blocks_in_memory to 4 or 2
4. Reduce video resolution or duration

### Model Loading Issues

- Ensure model weights are downloaded to `./weights/`
- Check that the config file exists in `./configs/`
- Verify CUDA is available: `torch.cuda.is_available()`

### Generation Errors

- For I2V: Make sure you've selected an I2V model (not T2V)
- For T2V: Make sure you've selected a T2V model (not I2V)
- Verify input image is provided for I2V generation

## Output

Generated videos are automatically saved to:
```
./outputs/t2v_YYYYMMDD_HHMMSS.mp4  (for T2V)
./outputs/i2v_YYYYMMDD_HHMMSS.mp4  (for I2V)
```

## Requirements

- Python 3.8+
- CUDA-capable GPU
- Gradio 4.0+
- All Kandinsky 5 dependencies

## Credits

- GUI styling inspired by [H1111](https://github.com/maybleMyers/H1111)
- Built with [Gradio](https://gradio.app/)
- Powered by [Kandinsky 5](https://github.com/ai-forever/Kandinsky-5)
