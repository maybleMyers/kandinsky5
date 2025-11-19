from math import floor, sqrt
from typing import Union

import transformers
import torch
import torchvision
import torchvision.transforms.functional as F
from torchvision.transforms import ToPILImage
from PIL import Image

from .generation_utils import generate_sample_i2v

torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose = True

MAX_AREA = 2048*2048
MAX_DIMENSION = 2048  # Maximum pixels per dimension to fit within RoPE max_pos=128

def log_vram_usage(stage_name, dit=None, vae=None, text_embedder=None):
    """Log VRAM usage and model locations for debugging."""
    if not torch.cuda.is_available():
        return

    # Get VRAM info
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    free, total = torch.cuda.mem_get_info()
    free_gb = free / 1024**3
    total_gb = total / 1024**3

    print(f"\n{'='*80}")
    print(f"VRAM USAGE AT: {stage_name}")
    print(f"{'='*80}")
    print(f"Allocated: {allocated:.2f} GB")
    print(f"Reserved:  {reserved:.2f} GB")
    print(f"Free:      {free_gb:.2f} GB / {total_gb:.2f} GB")

    # Check model locations
    print(f"\nModel Locations:")

    if dit is not None:
        if hasattr(dit, 'enable_block_swap') and dit.enable_block_swap:
            # Check DiT non-block components
            dit_device = next(dit.time_embeddings.parameters()).device
            print(f"  DiT (non-block components): {dit_device}")
            print(f"  DiT blocks in GPU: {len(dit._blocks_on_gpu) if hasattr(dit, '_blocks_on_gpu') else 'N/A'}")
            print(f"  DiT total blocks: {dit.num_visual_blocks if hasattr(dit, 'num_visual_blocks') else 'N/A'}")
        else:
            try:
                dit_device = next(dit.parameters()).device
                print(f"  DiT: {dit_device}")
            except:
                print(f"  DiT: Unable to determine device")

    if vae is not None:
        try:
            vae_device = next(vae.parameters()).device
            print(f"  VAE: {vae_device}")
        except:
            print(f"  VAE: Unable to determine device")

    if text_embedder is not None:
        try:
            # Check if text embedder models still exist
            if hasattr(text_embedder, 'embedder') and hasattr(text_embedder.embedder, 'model'):
                qwen_device = next(text_embedder.embedder.model.parameters()).device
                print(f"  Text Encoder (Qwen): {qwen_device}")
            else:
                print(f"  Text Encoder (Qwen): Deleted/Not loaded")

            if hasattr(text_embedder, 'clip_embedder') and hasattr(text_embedder.clip_embedder, 'model'):
                clip_device = next(text_embedder.clip_embedder.model.parameters()).device
                print(f"  Text Encoder (CLIP): {clip_device}")
            else:
                print(f"  Text Encoder (CLIP): Deleted/Not loaded")
        except:
            print(f"  Text Encoder: Unable to determine device")

    print(f"{'='*80}\n")

def resize_image(image, max_area, alignment=16):
    """
    Resize image to fit within limits while maintaining aspect ratio.
    Only downscales - never upscales. This respects user's resolution settings.

    Args:
        image: Input tensor image
        max_area: Maximum allowed area (height * width)
        alignment: Pixel alignment requirement (default 16, use 128 for NABLA attention)

    Note:
        For the 20B model with patch_size [1,2,2] and RoPE max_pos=128:
        - Max patches per dimension: 128
        - Max latent dimension: 128 * 2 = 256
        - Max pixel dimension: 256 * 8 = 2048 pixels
    """
    h, w = image.shape[2:]
    area = h * w

    # Check if we need to resize at all
    if area <= max_area and h <= MAX_DIMENSION and w <= MAX_DIMENSION:
        # Image is within limits, no resize needed
        return image, 1.0

    # Need to downscale - calculate scale factor
    k = sqrt(max_area / area) / alignment
    new_h = int(floor(h * k) * alignment)
    new_w = int(floor(w * k) * alignment)

    # Enforce per-dimension limit to stay within RoPE max_pos
    # RoPE3D has max_pos=(128, 128, 128) for (T, H, W) dimensions
    # With patch_size [1, 2, 2] and VAE 8x compression: max = 128 * 2 * 8 = 2048 pixels
    if new_h > MAX_DIMENSION or new_w > MAX_DIMENSION:
        # Scale down to fit within per-dimension limit
        scale_h = MAX_DIMENSION / new_h if new_h > MAX_DIMENSION else 1.0
        scale_w = MAX_DIMENSION / new_w if new_w > MAX_DIMENSION else 1.0
        scale = min(scale_h, scale_w)

        new_h = int(floor(new_h * scale / alignment) * alignment)
        new_w = int(floor(new_w * scale / alignment) * alignment)

        # Recalculate k for the final scale
        k = new_h / h

    return F.resize(image, (new_h, new_w)), k


def get_first_frame_from_image(image, vae, device, alignment=16):
    """
    Load and encode an image to latent space.

    Args:
        image: Path to image or PIL Image
        vae: VAE model
        device: Device to use
        alignment: Pixel alignment for resizing (use 128 for NABLA attention)
    """
    if isinstance(image, str):
        pil_image = Image.open(image).convert('RGB')
    elif isinstance(image, Image.Image):
        pil_image = image
    else:
        raise ValueError(f"unknown image type: {type(image)}")

    image = F.pil_to_tensor(pil_image).unsqueeze(0)
    image, k = resize_image(image, max_area=MAX_AREA, alignment=alignment)
    image = image / 128 - 1.  # KVAE m11 format: range [-1, 0.9921875]

    with torch.no_grad():
        # Use the VAE's dtype to avoid dtype mismatch
        vae_dtype = next(vae.parameters()).dtype
        image = image.to(device=device, dtype=vae_dtype).transpose(0, 1).unsqueeze(0)
        lat_image = vae.encode(image, opt_tiling=False).latent_dist.sample().squeeze(0).permute(1, 2, 3, 0)
        lat_image = lat_image * vae.config.scaling_factor

    return pil_image, lat_image, k


class Kandinsky5I2VPipeline:
    def __init__(
        self,
        device_map: Union[
            str, torch.device, dict
        ],  # {"dit": cuda:0, "vae": cuda:1, "text_embedder": cuda:1 }
        dit,
        text_embedder,
        vae,
        local_dit_rank: int = 0,
        world_size: int = 1,
        conf = None,
        offload: bool = False,
    ):
        self.dit = dit
        self.text_embedder = text_embedder
        self.vae = vae

        self.device_map = device_map
        self.local_dit_rank = local_dit_rank
        self.world_size = world_size
        self.conf = conf
        self.num_steps = conf.model.num_steps
        self.guidance_weight = conf.model.guidance_weight

        self.offload = offload


    def __call__(
        self,
        text: str,
        image: Union[str, Image.Image],
        time_length: int = 5,
        seed: int = None,
        num_steps: int = None,
        guidance_weight: float = None,
        scheduler_scale: float = 10.0,
        negative_caption: str = "Static, 2D cartoon, cartoon, 2d animation, paintings, images, worst quality, low quality, ugly, deformed, walking backwards",
        expand_prompts: bool = True,
        save_path: str = None,
        progress: bool = True,
        preview: int = None,
        preview_suffix: str = None,
    ):
        num_steps = self.num_steps if num_steps is None else num_steps
        guidance_weight = self.guidance_weight if guidance_weight is None else guidance_weight
        # SEED
        if seed is None:
            if self.local_dit_rank == 0:
                seed = torch.randint(2**32 - 1, (1,)).to(self.local_dit_rank)
            else:
                seed = torch.empty((1,), dtype=torch.int64).to(self.local_dit_rank)

            if self.world_size > 1:
                torch.distributed.broadcast(seed, 0)

            seed = seed.item()

        # PREPARATION
        num_frames = 1 if time_length == 0 else time_length * 24 // 4 + 1

        # For NABLA attention with fractal flattening, need 128-pixel alignment
        # For other attention types, 16-pixel alignment is sufficient
        try:
            attention_type = self.conf.model.attention.type
        except (AttributeError, KeyError):
            attention_type = 'flash'  # Default to flash if attention config is missing
        alignment = 128 if attention_type == 'nabla' else 16

        # Load VAE for encoding if using offload or block swap
        force_offload = hasattr(self.dit, 'enable_block_swap') and self.dit.enable_block_swap

        # Log VRAM before VAE encoding
        log_vram_usage("BEFORE VAE ENCODE (I2V)", dit=self.dit, vae=self.vae, text_embedder=self.text_embedder)

        if self.offload or force_offload:
            self.vae = self.vae.to(self.device_map["vae"], non_blocking=True)

        image, image_lat, k = get_first_frame_from_image(image, self.vae, self.device_map["vae"], alignment=alignment)

        # Log VRAM after VAE encoding, before offload
        log_vram_usage("AFTER VAE ENCODE, BEFORE OFFLOAD (I2V)", dit=self.dit, vae=self.vae, text_embedder=self.text_embedder)

        if self.offload or force_offload:
            self.vae = self.vae.to("cpu", non_blocking=True)
            torch.cuda.empty_cache()

        # Log VRAM after VAE offload
        log_vram_usage("AFTER VAE OFFLOAD (I2V)", dit=self.dit, vae=self.vae, text_embedder=self.text_embedder)

        caption = text
        if expand_prompts:
            transformers.set_seed(seed)
            if self.local_dit_rank == 0:
                # Load text embedder if using offload or block swap (which keeps models on CPU initially)
                force_offload = hasattr(self.dit, 'enable_block_swap') and self.dit.enable_block_swap
                if self.offload or force_offload:
                    self.text_embedder = self.text_embedder.to(self.device_map["text_embedder"])
                caption = self.text_embedder.embedder.expand_text_prompt(caption, image, device=self.device_map["text_embedder"])
            if self.world_size > 1:
                caption = [caption]
                torch.distributed.broadcast_object_list(caption, 0)
                caption = caption[0]

        height, width = image_lat.shape[1:3]
        shape = (1, num_frames, height, width, 16)

        previewer = None
        if preview is not None and preview > 0:
            print(f"\n>>> i2v_pipeline: Initializing previewer with preview={preview}")
            try:
                from scripts.latentpreviewer import LatentPreviewer
                import os

                g_temp = torch.Generator(device=self.device_map["dit"])
                g_temp.manual_seed(seed)
                initial_latent = torch.randn(shape[0] * shape[1], shape[2], shape[3], shape[4], device=self.device_map["dit"], generator=g_temp)
                print(f">>> initial_latent shape before permute: {initial_latent.shape}")
                initial_latent = initial_latent.permute(3, 0, 1, 2)
                print(f">>> initial_latent shape after permute: {initial_latent.shape}")

                timesteps = torch.linspace(1, 0, num_steps + 1, device=self.device_map["dit"])
                timesteps = scheduler_scale * timesteps / (1 + (scheduler_scale - 1) * timesteps)
                timesteps = timesteps[:-1] * 1000
                print(f">>> timesteps shape: {timesteps.shape}")

                class Args:
                    def __init__(self, save_path, fps):
                        self.save_path = save_path
                        self.fps = fps

                args_obj = Args(
                    save_path=os.path.dirname(save_path) if save_path else './',
                    fps=24
                )

                previewer = LatentPreviewer(
                    args=args_obj,
                    original_latents=initial_latent,
                    timesteps=timesteps,
                    device=self.device_map["dit"],
                    dtype=torch.bfloat16,
                    model_type="hunyuan"
                )
                print(f">>> i2v_pipeline: Previewer initialized successfully, will generate preview every {preview} steps")
            except Exception as e:
                print(f">>> i2v_pipeline: Failed to initialize previewer: {e}")
                import traceback
                traceback.print_exc()
                previewer = None
        else:
            print(f">>> i2v_pipeline: Preview disabled (preview={preview})")

        force_offload = hasattr(self.dit, 'enable_block_swap') and self.dit.enable_block_swap
        images = generate_sample_i2v(
            shape,
            caption,
            self.dit,
            self.vae,
            self.conf,
            text_embedder=self.text_embedder,
            images=image_lat,
            num_steps=num_steps,
            guidance_weight=guidance_weight,
            scheduler_scale=scheduler_scale,
            negative_caption=negative_caption,
            seed=seed,
            device=self.device_map["dit"],
            vae_device=self.device_map["vae"],
            progress=progress,
            offload=self.offload,
            force_offload=force_offload,
            previewer=previewer,
            preview_interval=preview,
            preview_suffix=preview_suffix,
        )

        # Delete text encoder to free RAM - it's no longer needed
        del self.text_embedder
        torch.cuda.empty_cache()
        import gc
        gc.collect()

        if k > 16:
            h, w = images.shape[-2:]
            images = F.resize(images[0], (int(h / k / 16), int(w / k / 16)))

        # RESULTS
        if self.local_dit_rank == 0:
            if time_length == 0:
                return_images = []
                for image in images.squeeze(2).cpu():
                    return_images.append(ToPILImage()(image))
                if save_path is not None:
                    if isinstance(save_path, str):
                        save_path = [save_path]
                    if len(save_path) == len(return_images):
                        for path, image in zip(save_path, return_images):
                            image.save(path)
                return return_images
            else:
                if save_path is not None:
                    if isinstance(save_path, str):
                        save_path = [save_path]
                    if len(save_path) == len(images):
                        for path, video in zip(save_path, images):
                            torchvision.io.write_video(
                                path,
                                video.float().permute(1, 2, 3, 0).cpu().numpy(),
                                fps=24,
                                options={"crf": "5"},
                            )
                return images
