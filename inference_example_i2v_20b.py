"""
Example script for running the 20B Kandinsky I2V Pro model with block swapping.

This script demonstrates how to use the block swapping feature to run the large
20B parameter model on GPUs with limited VRAM (e.g., 48GB).

Requirements:
- NVIDIA GPU with 48GB+ VRAM (A6000, RTX 6000 Ada, or better)
- All model weights downloaded to ./weights/
- Config file: configs/config_5s_i2v_pro_20b.yaml
"""

import torch
from kandinsky import get_I2V_pipeline_with_block_swap

# Configuration
device_map = {
    "dit": torch.device('cuda:0'),
    "vae": torch.device('cuda:0'),
    "text_embedder": torch.device('cuda:0')
}

# Load pipeline with block swapping
# The config file specifies blocks_in_memory=6, which keeps 6 out of 60
# transformer blocks in GPU memory at a time
print("Loading 20B I2V Pro model with block swapping...")
pipe = get_I2V_pipeline_with_block_swap(
    device_map=device_map,
    conf_path="./configs/config_5s_i2v_pro_20b.yaml",

    # Optional: Override config settings
    # blocks_in_memory=6,        # Keep 6 blocks in GPU memory (adjust for your VRAM)
    # enable_block_swap=True,    # Enable block swapping

    # Memory optimization options
    offload=False,               # Set True to offload VAE/text encoder when not in use
    quantized_qwen=True,         # Use 4-bit quantized Qwen to save ~12GB VRAM

    # Attention engine (auto selects best available)
    attention_engine="auto",     # Options: auto, flash_attention_2, flash_attention_3, sdpa, sage
)

print("Model loaded successfully!")

# Generate video
print("Generating video...")
result = pipe(
    text="The Dragon breaths fire.",
    image="./assets/test_image.jpg",
    time_length=5,                # 5 second video
    seed=137,
    num_steps=50,                 # 50 diffusion steps
    save_path='./test_20b.mp4',
)

print(f"Video generated and saved to: ./test_20b.mp4")

# Memory usage tips:
#
# If you run out of VRAM, try:
# 1. Reduce blocks_in_memory (e.g., from 6 to 4 or even 3)
# 2. Enable offload=True
# 3. Use quantized_qwen=True (if not already)
# 4. Reduce video resolution or duration
#
# Example for 40GB VRAM:
#   blocks_in_memory=4, offload=True, quantized_qwen=True
#
# Example for 24GB VRAM:
#   blocks_in_memory=2, offload=True, quantized_qwen=True
#   (will be slower due to more swapping)
