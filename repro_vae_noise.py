
import torch
import os
from PIL import Image
import torchvision.transforms.functional as F
from kandinsky.utils import build_vae
from omegaconf import OmegaConf
import numpy as np

def test_vae_noise():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load VAE
    # We need a config. Let's create a minimal one or load the existing one.
    # The existing config is at configs/config_5s_i2v_pro_20b.yaml
    config_path = "configs/config_5s_i2v_pro_20b.yaml"
    if not os.path.exists(config_path):
        print(f"Config not found at {config_path}")
        return

    conf = OmegaConf.load(config_path)
    
    # We need to ensure weights are downloaded/available. 
    # Assuming they are since the user is running test.py.
    # We'll use the build_vae from utils.
    
    print("Building VAE...")
    try:
        vae = build_vae(conf.model.vae, dtype=torch.float32) # Use float32 for precision check
        vae = vae.to(device).eval()
    except Exception as e:
        print(f"Failed to build VAE: {e}")
        return

    # 2. Create a dummy image or load one
    # Create a simple gradient image
    width, height = 512, 512
    image = Image.new('RGB', (width, height))
    for x in range(width):
        for y in range(height):
            image.putpixel((x, y), (int(x/width*255), int(y/height*255), 0))
    
    # 3. Preprocess image (same as i2v_pipeline)
    img_tensor = F.pil_to_tensor(image).unsqueeze(0).float()
    # Normalization from i2v_pipeline
    img_tensor = img_tensor / 128 - 1. 
    img_tensor = img_tensor.to(device)
    
    # Transpose for VAE: (B, C, T, H, W) -> T=1
    img_tensor = img_tensor.unsqueeze(2) 
    
    print(f"Input tensor shape: {img_tensor.shape}")
    print(f"Input tensor range: [{img_tensor.min().item()}, {img_tensor.max().item()}]")

    # 4. Encode using sample() multiple times
    print("\nTesting sample() variance...")
    latents_samples = []
    for i in range(5):
        with torch.no_grad():
            # encode returns AutoencoderKLOutput
            # We need to access latent_dist
            posterior = vae.encode(img_tensor).latent_dist
            sample = posterior.sample()
            latents_samples.append(sample)
            
            # Decode to check visual quality
            decoded = vae.decode(sample).sample
            # Post-process
            decoded = ((decoded.clamp(-1.0, 127/128) + 1.0) * 128).to(torch.uint8)
            # Save first sample
            if i == 0:
                # (B, C, T, H, W) -> (C, H, W)
                decoded_img = decoded[0, :, 0, :, :].cpu()
                decoded_pil = F.to_pil_image(decoded_img)
                decoded_pil.save("repro_sample_0.png")
                print("Saved repro_sample_0.png")

    # Calculate variance between samples
    stack = torch.stack(latents_samples)
    variance = torch.var(stack, dim=0).mean().item()
    print(f"Mean variance between 5 samples: {variance}")
    
    if variance > 1e-5:
        print(">> CONFIRMED: sample() introduces significant noise.")
    else:
        print(">> RESULT: sample() variance is low (unexpected).")

    # 5. Encode using mode()
    print("\nTesting mode()...")
    try:
        with torch.no_grad():
            posterior = vae.encode(img_tensor).latent_dist
            mode = posterior.mode()
            
            # Decode
            decoded_mode = vae.decode(mode).sample
            decoded_mode = ((decoded_mode.clamp(-1.0, 127/128) + 1.0) * 128).to(torch.uint8)
            
            decoded_mode_img = decoded_mode[0, :, 0, :, :].cpu()
            decoded_mode_pil = F.to_pil_image(decoded_mode_img)
            decoded_mode_pil.save("repro_mode.png")
            print("Saved repro_mode.png")
            
            # Check consistency
            mode2 = posterior.mode()
            diff = (mode - mode2).abs().max().item()
            print(f"Difference between two mode() calls: {diff}")
            
            if diff < 1e-6:
                print(">> CONFIRMED: mode() is deterministic.")
            
    except AttributeError:
        print("DiagonalGaussianDistribution does not have mode() method.")
    except Exception as e:
        print(f"Error testing mode(): {e}")

if __name__ == "__main__":
    test_vae_noise()
