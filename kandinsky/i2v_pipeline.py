from math import floor, sqrt
from typing import Union, Tuple, Optional
from contextlib import contextmanager, nullcontext

import transformers
import torch
import torchvision
import torchvision.transforms.functional as F
from torchvision.transforms import ToPILImage
from PIL import Image

from .generation_utils import generate_sample_i2v, generate_sample_v2v, generate_sample_denoise
from .models import vae_v2v

torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose = True

MAX_AREA = 2048*2048
MAX_DIMENSION = 2048  # Maximum pixels per dimension to fit within RoPE max_pos=128

# Memory limit for chunked operations (from musubi-tuner)
# Reduce if you see OOM during encoding
MEMORY_LIMIT_MB = 512


# =============================================================================
# V2V-safe tiling context manager
# =============================================================================

@contextmanager
def v2v_safe_tiling(vae):
    """
    Context manager that temporarily replaces VAE's tiled_encode/decode methods
    with v2v-safe versions that ensure full edge coverage, and forces spatial
    tiling to avoid OOM during v2v encoding.

    The v2v versions add extra edge tiles when the image dimensions don't align
    perfectly with the tile stride, preventing loss of edge pixels.

    Usage:
        with v2v_safe_tiling(vae):
            latents = vae.encode(video_tensor, opt_tiling=True).latent_dist.mode()
    """
    # Get the v2v-safe methods from the AutoencoderKLHunyuanVideo class in vae_v2v
    v2v_vae_class = vae_v2v.AutoencoderKLHunyuanVideo

    # Save original methods
    original_tiled_encode = vae.tiled_encode
    original_tiled_decode = vae.tiled_decode

    # Save original tile sizes (get_enc_optimal_tiling can set these to full resolution,
    # which bypasses spatial tiling and causes OOM during v2v)
    original_tile_min_height = vae.tile_sample_min_height
    original_tile_min_width = vae.tile_sample_min_width
    original_tile_stride_height = vae.tile_sample_stride_height
    original_tile_stride_width = vae.tile_sample_stride_width

    try:
        # Replace with v2v-safe methods (bound to this vae instance)
        import types
        vae.tiled_encode = types.MethodType(v2v_vae_class.tiled_encode, vae)
        vae.tiled_decode = types.MethodType(v2v_vae_class.tiled_decode, vae)

        # Force conservative tile sizes to ensure spatial tiling is always used
        # This prevents OOM when get_enc_optimal_tiling sets tiles to full resolution
        vae.tile_sample_min_height = 256
        vae.tile_sample_min_width = 256
        vae.tile_sample_stride_height = 192
        vae.tile_sample_stride_width = 192

        # Set flag to skip tile setup in encode() so our settings aren't overwritten
        vae._skip_tile_setup = True

        yield vae
    finally:
        # Restore original methods
        vae.tiled_encode = original_tiled_encode
        vae.tiled_decode = original_tiled_decode

        # Restore original tile sizes
        vae.tile_sample_min_height = original_tile_min_height
        vae.tile_sample_min_width = original_tile_min_width
        vae.tile_sample_stride_height = original_tile_stride_height
        vae.tile_sample_stride_width = original_tile_stride_width

        # Reset the skip flag
        vae._skip_tile_setup = False


# =============================================================================
# Musubi-tuner style blending functions for seamless tiling
# =============================================================================

def blend_v(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
    """
    Blend tensor b vertically into a at blend_extent region.
    Operates on tensors with shape [B, C, T, H, W].
    From musubi-tuner: hunyuan_video_1_5_vae.py
    """
    blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
    if blend_extent <= 0:
        return b
    b = b.clone()
    for y in range(blend_extent):
        weight = y / blend_extent
        b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - weight) + b[:, :, :, y, :] * weight
    return b


def blend_h(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
    """
    Blend tensor b horizontally into a at blend_extent region.
    Operates on tensors with shape [B, C, T, H, W].
    From musubi-tuner: hunyuan_video_1_5_vae.py
    """
    blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
    if blend_extent <= 0:
        return b
    b = b.clone()
    for x in range(blend_extent):
        weight = x / blend_extent
        b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - weight) + b[:, :, :, :, x] * weight
    return b


def blend_t(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
    """
    Blend tensor b temporally into a at blend_extent region.
    Operates on tensors with shape [B, C, T, H, W].
    From musubi-tuner: hunyuan_video_1_5_vae.py
    """
    blend_extent = min(a.shape[2], b.shape[2], blend_extent)
    if blend_extent <= 0:
        return b
    b = b.clone()
    for t in range(blend_extent):
        weight = t / blend_extent
        b[:, :, t, :, :] = a[:, :, -blend_extent + t, :, :] * (1 - weight) + b[:, :, t, :, :] * weight
    return b


def encode_latent_deterministic(
    vae,
    input_tensor: torch.Tensor,
    device: torch.device,
    use_tiling: bool = False,
) -> torch.Tensor:
    """
    Encode input tensor to latent space using deterministic .mode() sampling.
    This provides more consistent results than stochastic .sample().

    From musubi-tuner: Uses mode() for deterministic encoding.

    Args:
        vae: VAE model
        input_tensor: Input tensor [B, C, T, H, W]
        device: Device to use
        use_tiling: Whether to use tiled encoding for large inputs

    Returns:
        Latent tensor scaled by VAE scaling factor
    """
    with torch.no_grad():
        vae_dtype = next(vae.parameters()).dtype
        input_tensor = input_tensor.to(device=device, dtype=vae_dtype)

        # Use opt_tiling based on input size
        latent_dist = vae.encode(input_tensor, opt_tiling=use_tiling).latent_dist

        # Use .mode() for deterministic encoding (musubi-tuner approach)
        # This gives the mean of the latent distribution rather than sampling
        latent = latent_dist.mode()

        # Scale by VAE scaling factor
        latent = latent * vae.config.scaling_factor

    return latent


def spatial_tiled_encode(
    vae,
    x: torch.Tensor,
    tile_size: int = 256,
    tile_overlap: float = 0.25,
) -> torch.Tensor:
    """
    Encode large images/videos using spatial tiling with smooth blending.
    From musubi-tuner: spatial_tiled_encode in hunyuan_video_1_5_vae.py

    Args:
        vae: VAE model
        x: Input tensor [B, C, T, H, W]
        tile_size: Size of each tile in pixels
        tile_overlap: Overlap factor between tiles (0.0 to 0.5)

    Returns:
        Encoded latent tensor
    """
    B, C, T, H, W = x.shape
    spatial_compression = vae.spatial_compression_ratio  # Usually 8

    # Calculate tile parameters
    overlap_size = int(tile_size * (1 - tile_overlap))
    blend_extent_latent = int((tile_size // spatial_compression) * tile_overlap)
    tile_limit_latent = (tile_size // spatial_compression) - blend_extent_latent

    rows = []
    for i in range(0, H, overlap_size):
        row = []
        for j in range(0, W, overlap_size):
            # Extract tile
            tile = x[:, :, :, i:i + tile_size, j:j + tile_size]

            # Encode tile
            with torch.no_grad():
                tile_latent = vae.encode(tile, opt_tiling=False).latent_dist.mode()
                tile_latent = tile_latent * vae.config.scaling_factor

            row.append(tile_latent)
        rows.append(row)

    # Blend tiles together
    result_rows = []
    for i, row in enumerate(rows):
        result_row = []
        for j, tile in enumerate(row):
            if i > 0:
                tile = blend_v(rows[i - 1][j], tile, blend_extent_latent)
            if j > 0:
                tile = blend_h(row[j - 1], tile, blend_extent_latent)
            result_row.append(tile[:, :, :, :tile_limit_latent, :tile_limit_latent])
        result_rows.append(torch.cat(result_row, dim=-1))

    latent = torch.cat(result_rows, dim=-2)
    return latent


def get_conditioning_latents_from_two_images(start_image, end_image, vae, device, alignment=16, use_deterministic=True):
    """
    Encode two images to latent space for image-to-image-video generation.
    Mirrors get_conditioning_frames_from_two_videos() but for single images.

    Args:
        start_image: Path to starting image or PIL Image
        end_image: Path to ending image or PIL Image
        vae: VAE model
        device: Device to use
        alignment: Pixel alignment for resizing (16 standard, 128 for NABLA)
        use_deterministic: If True, use .mode() for deterministic encoding (musubi-tuner style).
                          If False, use .sample() for stochastic encoding.

    Returns:
        Tuple of (start_latent, end_latent, scale_factor)
        start_latent: Tensor of shape [1, H, W, C] - single frame latent
        end_latent: Tensor of shape [1, H, W, C] - single frame latent
        scale_factor: Scale factor applied during resizing
    """
    from PIL import Image

    # Load start image
    if isinstance(start_image, str):
        start_pil = Image.open(start_image).convert('RGB')
    elif isinstance(start_image, Image.Image):
        start_pil = start_image
    else:
        raise ValueError(f"Unknown start_image type: {type(start_image)}")

    # Load end image
    if isinstance(end_image, str):
        end_pil = Image.open(end_image).convert('RGB')
    elif isinstance(end_image, Image.Image):
        end_pil = end_image
    else:
        raise ValueError(f"Unknown end_image type: {type(end_image)}")

    # Process start image - determines target dimensions
    start_tensor = F.pil_to_tensor(start_pil).unsqueeze(0)
    start_tensor, scale_factor = resize_image(start_tensor, max_area=MAX_AREA, alignment=alignment)
    target_h, target_w = start_tensor.shape[2], start_tensor.shape[3]

    # Process end image - resize to match start image dimensions
    end_tensor = F.pil_to_tensor(end_pil).unsqueeze(0)
    import torch.nn.functional as F_torch
    end_tensor = F_torch.interpolate(
        end_tensor.float(),
        size=(target_h, target_w),
        mode='bilinear',
        align_corners=False
    )

    # Normalize to [-1, 1]
    start_tensor = start_tensor / 127.5 - 1.
    end_tensor = end_tensor / 127.5 - 1.

    # Encode through VAE
    with torch.no_grad():
        vae_dtype = next(vae.parameters()).dtype

        # Encode start image
        # Input shape: [1, C, H, W] -> transpose to [C, 1, H, W] -> unsqueeze to [1, C, 1, H, W]
        start_input = start_tensor.to(device=device, dtype=vae_dtype).transpose(0, 1).unsqueeze(0)
        start_latent_dist = vae.encode(start_input, opt_tiling=False).latent_dist
        # Use .mode() for deterministic or .sample() for stochastic (musubi-tuner uses mode)
        start_latent = start_latent_dist.mode() if use_deterministic else start_latent_dist.sample()
        start_latent = start_latent.squeeze(0).permute(1, 2, 3, 0) * vae.config.scaling_factor
        # Output shape: [1, H_latent, W_latent, C_latent]

        # Encode end image
        end_input = end_tensor.to(device=device, dtype=vae_dtype).transpose(0, 1).unsqueeze(0)
        end_latent_dist = vae.encode(end_input, opt_tiling=False).latent_dist
        end_latent = end_latent_dist.mode() if use_deterministic else end_latent_dist.sample()
        end_latent = end_latent.squeeze(0).permute(1, 2, 3, 0) * vae.config.scaling_factor

    return start_latent, end_latent, scale_factor

def extract_last_frames_from_video(video_path, num_frames, target_fps=24):
    """
    Extract the last N frames from a video file.

    Args:
        video_path: Path to the video file
        num_frames: Number of frames to extract from the end
        target_fps: Target FPS for frame extraction (default 24)

    Returns:
        List of PIL Images (last N frames)
    """
    import av
    import numpy as np

    container = av.open(video_path)
    video_stream = container.streams.video[0]

    # Get video properties
    video_fps = float(video_stream.average_rate)
    total_frames = video_stream.frames

    # Calculate frame indices to extract
    # Downsample to target_fps if video fps is higher
    if video_fps > target_fps:
        frame_skip = int(video_fps / target_fps)
    else:
        frame_skip = 1

    # Decode all frames and keep last N
    frames = []
    for frame in container.decode(video=0):
        if len(frames) >= num_frames * frame_skip:
            frames.pop(0)
        frames.append(frame)

    container.close()

    # Apply frame skip to get target fps
    frames = frames[::frame_skip][-num_frames:]

    # Convert to PIL Images
    pil_frames = []
    for frame in frames:
        img = frame.to_image()
        pil_frames.append(img.convert('RGB'))

    return pil_frames


def extract_first_frames_from_video(video_path, num_frames, target_fps=24):
    """
    Extract the first N frames from a video file.

    Args:
        video_path: Path to the video file
        num_frames: Number of frames to extract from the beginning
        target_fps: Target FPS for frame extraction (default 24)

    Returns:
        List of PIL Images (first N frames)
    """
    import av
    import numpy as np

    container = av.open(video_path)
    video_stream = container.streams.video[0]

    # Get video properties
    video_fps = float(video_stream.average_rate)

    # Calculate frame skip for downsampling to target_fps if needed
    if video_fps > target_fps:
        frame_skip = int(video_fps / target_fps)
    else:
        frame_skip = 1

    # Decode and keep first N frames (accounting for frame skip)
    frames = []
    frame_count = 0
    for frame in container.decode(video=0):
        if frame_count % frame_skip == 0:
            frames.append(frame)
            if len(frames) >= num_frames:
                break
        frame_count += 1

    container.close()

    # Convert to PIL Images
    pil_frames = []
    for frame in frames:
        img = frame.to_image()
        pil_frames.append(img.convert('RGB'))

    return pil_frames


def get_conditioning_frames_from_video(video_path, num_frames, vae, device, alignment=16, use_deterministic=True):
    """
    Load video and encode last N frames to latent space for video continuation.

    Args:
        video_path: Path to the video file
        num_frames: Number of conditioning frames to extract
        vae: VAE model
        device: Device to use
        alignment: Pixel alignment for resizing
        use_deterministic: If True, use .mode() for deterministic encoding (musubi-tuner style).

    Returns:
        Tuple of (latents, scale_factor)
        latents: Tensor of shape [num_frames, H, W, C]
    """
    # Extract frames from video
    pil_frames = extract_last_frames_from_video(video_path, num_frames)

    if len(pil_frames) < num_frames:
        raise ValueError(f"Video has only {len(pil_frames)} frames, need {num_frames}")

    # Convert frames to tensors and encode
    latents_list = []
    scale_factor = None

    for i, pil_image in enumerate(pil_frames):
        # Convert to tensor
        image = F.pil_to_tensor(pil_image).unsqueeze(0)
        image, k = resize_image(image, max_area=MAX_AREA, alignment=alignment)
        image = image / 127.5 - 1.

        if scale_factor is None:
            scale_factor = k

        with torch.no_grad():
            vae_dtype = next(vae.parameters()).dtype
            image = image.to(device=device, dtype=vae_dtype).transpose(0, 1).unsqueeze(0)
            latent_dist = vae.encode(image, opt_tiling=False).latent_dist
            # Use .mode() for deterministic or .sample() for stochastic (musubi-tuner uses mode)
            lat_image = latent_dist.mode() if use_deterministic else latent_dist.sample()
            lat_image = lat_image.squeeze(0).permute(1, 2, 3, 0) * vae.config.scaling_factor
            latents_list.append(lat_image)

    # Stack latents: [num_frames, H, W, C]
    latents = torch.cat(latents_list, dim=0)

    return latents, scale_factor


def get_conditioning_frames_from_two_videos(video1_path, video2_path, num_frames, vae, device, alignment=16, use_deterministic=True):
    """
    Load two videos and encode conditioning frames for video joining:
    - Last N frames from video1 (start conditioning)
    - First N frames from video2 (end conditioning)

    Args:
        video1_path: Path to the first video file
        video2_path: Path to the second video file
        num_frames: Number of conditioning frames to extract from each video
        vae: VAE model
        device: Device to use
        alignment: Pixel alignment for resizing
        use_deterministic: If True, use .mode() for deterministic encoding (musubi-tuner style).

    Returns:
        Tuple of (start_latents, end_latents, scale_factor)
        start_latents: Tensor of shape [num_frames, H, W, C] from video1
        end_latents: Tensor of shape [num_frames, H, W, C] from video2
    """
    import torch.nn.functional as F_torch

    # Extract last frames from video1 and first frames from video2
    pil_frames_start = extract_last_frames_from_video(video1_path, num_frames)
    pil_frames_end = extract_first_frames_from_video(video2_path, num_frames)

    if len(pil_frames_start) < num_frames:
        raise ValueError(f"Video1 has only {len(pil_frames_start)} frames, need {num_frames}")
    if len(pil_frames_end) < num_frames:
        raise ValueError(f"Video2 has only {len(pil_frames_end)} frames, need {num_frames}")

    # Determine target size from first frame of video1
    first_image = F.pil_to_tensor(pil_frames_start[0]).unsqueeze(0)
    first_image, scale_factor = resize_image(first_image, max_area=MAX_AREA, alignment=alignment)
    target_h, target_w = first_image.shape[2], first_image.shape[3]

    vae_dtype = next(vae.parameters()).dtype

    # Encode start frames (from video1)
    start_latents_list = []
    for i, pil_image in enumerate(pil_frames_start):
        image = F.pil_to_tensor(pil_image).unsqueeze(0)
        image, _ = resize_image(image, max_area=MAX_AREA, alignment=alignment)
        image = image / 127.5 - 1.

        with torch.no_grad():
            image = image.to(device=device, dtype=vae_dtype).transpose(0, 1).unsqueeze(0)
            latent_dist = vae.encode(image, opt_tiling=False).latent_dist
            # Use .mode() for deterministic or .sample() for stochastic (musubi-tuner uses mode)
            lat_image = latent_dist.mode() if use_deterministic else latent_dist.sample()
            lat_image = lat_image.squeeze(0).permute(1, 2, 3, 0) * vae.config.scaling_factor
            start_latents_list.append(lat_image)

    # Encode end frames (from video2) - resize to match video1 dimensions
    end_latents_list = []
    for i, pil_image in enumerate(pil_frames_end):
        image = F.pil_to_tensor(pil_image).unsqueeze(0)
        # Resize to match video1 dimensions
        image = F_torch.interpolate(image.float(), size=(target_h, target_w), mode='bilinear', align_corners=False)
        image = image / 127.5 - 1.

        with torch.no_grad():
            image = image.to(device=device, dtype=vae_dtype).transpose(0, 1).unsqueeze(0)
            latent_dist = vae.encode(image, opt_tiling=False).latent_dist
            lat_image = latent_dist.mode() if use_deterministic else latent_dist.sample()
            lat_image = lat_image.squeeze(0).permute(1, 2, 3, 0) * vae.config.scaling_factor
            end_latents_list.append(lat_image)

    # Stack latents: [num_frames, H, W, C]
    start_latents = torch.cat(start_latents_list, dim=0)
    end_latents = torch.cat(end_latents_list, dim=0)

    return start_latents, end_latents, scale_factor


def get_conditioning_video_and_image(video_path, end_image, num_frames, vae, device, alignment=16, use_deterministic=True):
    """
    Load a video and an end image for video + end image joining mode:
    - Last N frames from video (start conditioning)
    - Single end image replicated to N frames (end conditioning)

    Args:
        video_path: Path to the video file
        end_image: Path to ending image or PIL Image
        num_frames: Number of conditioning frames to extract from video
        vae: VAE model
        device: Device to use
        alignment: Pixel alignment for resizing
        use_deterministic: If True, use .mode() for deterministic encoding.

    Returns:
        Tuple of (start_latents, end_latents, scale_factor)
        start_latents: Tensor of shape [num_frames, H, W, C] from video
        end_latents: Tensor of shape [num_frames, H, W, C] from end image (replicated)
    """
    import torch.nn.functional as F_torch
    from PIL import Image

    # Extract last frames from video
    pil_frames_start = extract_last_frames_from_video(video_path, num_frames)

    if len(pil_frames_start) < num_frames:
        raise ValueError(f"Video has only {len(pil_frames_start)} frames, need {num_frames}")

    # Load end image
    if isinstance(end_image, str):
        end_pil = Image.open(end_image).convert('RGB')
    elif isinstance(end_image, Image.Image):
        end_pil = end_image
    else:
        raise ValueError(f"Unknown end_image type: {type(end_image)}")

    # Determine target size from first frame of video
    first_image = F.pil_to_tensor(pil_frames_start[0]).unsqueeze(0)
    first_image, scale_factor = resize_image(first_image, max_area=MAX_AREA, alignment=alignment)
    target_h, target_w = first_image.shape[2], first_image.shape[3]

    vae_dtype = next(vae.parameters()).dtype

    # Encode start frames (from video)
    start_latents_list = []
    for i, pil_image in enumerate(pil_frames_start):
        image = F.pil_to_tensor(pil_image).unsqueeze(0)
        image, _ = resize_image(image, max_area=MAX_AREA, alignment=alignment)
        image = image / 127.5 - 1.

        with torch.no_grad():
            image = image.to(device=device, dtype=vae_dtype).transpose(0, 1).unsqueeze(0)
            latent_dist = vae.encode(image, opt_tiling=False).latent_dist
            lat_image = latent_dist.mode() if use_deterministic else latent_dist.sample()
            lat_image = lat_image.squeeze(0).permute(1, 2, 3, 0) * vae.config.scaling_factor
            start_latents_list.append(lat_image)

    # Encode end image - resize to match video dimensions
    end_image_tensor = F.pil_to_tensor(end_pil).unsqueeze(0)
    end_image_tensor = F_torch.interpolate(end_image_tensor.float(), size=(target_h, target_w), mode='bilinear', align_corners=False)
    end_image_tensor = end_image_tensor / 127.5 - 1.

    with torch.no_grad():
        end_image_tensor = end_image_tensor.to(device=device, dtype=vae_dtype).transpose(0, 1).unsqueeze(0)
        latent_dist = vae.encode(end_image_tensor, opt_tiling=False).latent_dist
        end_lat_image = latent_dist.mode() if use_deterministic else latent_dist.sample()
        end_lat_image = end_lat_image.squeeze(0).permute(1, 2, 3, 0) * vae.config.scaling_factor

    # Replicate end image latent to match num_frames
    end_latents_list = [end_lat_image for _ in range(num_frames)]

    # Stack latents: [num_frames, H, W, C]
    start_latents = torch.cat(start_latents_list, dim=0)
    end_latents = torch.cat(end_latents_list, dim=0)

    return start_latents, end_latents, scale_factor


def encode_video_to_latents(video_path, vae, device, alignment=16, target_fps=24, max_frames=None, use_deterministic=True, v2v_mode=False):
    """
    Load a video file and encode to latent space with proper temporal compression.

    The VAE has 4x temporal compression, so 17 video frames -> 5 latent frames.
    Videos are processed in small chunks to avoid OOM.

    Uses musubi-tuner style encoding with:
    - Deterministic .mode() sampling for consistent results
    - Memory-efficient chunked processing for high-res/long videos
    - Proper temporal blending at chunk boundaries

    Args:
        video_path: Path to the video file
        vae: VAE model
        device: Device to use
        alignment: Pixel alignment for resizing
        target_fps: Target FPS for frame extraction
        max_frames: Maximum number of video frames to process (None = all frames)
        use_deterministic: If True, use .mode() for deterministic encoding (musubi-tuner style).
                          If False, use .sample() for stochastic encoding.
        v2v_mode: If True, use v2v-safe tiling that ensures full edge coverage.
                  This prevents loss of edge pixels when dimensions don't align with tile stride.

    Returns:
        Tuple of (latents, scale_factor, num_video_frames)
        latents: Tensor of shape [num_latent_frames, H, W, C]
    """
    import av
    import numpy as np
    import torch.nn.functional as F_torch

    container = av.open(video_path)
    video_stream = container.streams.video[0]

    # Get video properties
    video_fps = float(video_stream.average_rate)

    # Calculate frame skip for downsampling to target_fps
    if video_fps > target_fps:
        frame_skip = int(video_fps / target_fps)
    else:
        frame_skip = 1

    # Decode all frames to PIL (memory efficient)
    pil_frames = []
    frame_count = 0
    for frame in container.decode(video=0):
        if frame_count % frame_skip == 0:
            img = frame.to_image()
            pil_frames.append(img.convert('RGB'))
            if max_frames is not None and len(pil_frames) >= max_frames:
                break
        frame_count += 1

    container.close()

    if len(pil_frames) == 0:
        raise ValueError(f"No frames extracted from video: {video_path}")

    num_video_frames = len(pil_frames)
    print(f">>> Extracted {num_video_frames} video frames", flush=True)

    # Determine target size from first frame
    first_image = F.pil_to_tensor(pil_frames[0]).unsqueeze(0)
    first_image, scale_factor = resize_image(first_image, max_area=MAX_AREA, alignment=alignment)
    target_h, target_w = first_image.shape[2], first_image.shape[3]

    vae_dtype = next(vae.parameters()).dtype

    # Calculate expected latent frames for full video
    expected_latent_frames = (num_video_frames - 1) // 4 + 1

    sampling_mode = "deterministic (.mode())" if use_deterministic else "stochastic (.sample())"
    print(f">>> Encoding video to latent space (4x temporal compression, {sampling_mode})...", flush=True)
    print(f">>> Resolution: {target_w}x{target_h}, {num_video_frames} frames -> {expected_latent_frames} latents expected", flush=True)

    # Estimate memory requirements to choose encoding strategy
    # High-res or long videos need chunked encoding from the start
    pixels_per_frame = target_h * target_w
    total_pixels = pixels_per_frame * num_video_frames
    free_mem = torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else 0

    # Heuristic: if resolution > 768x512 and video > 5s, use chunked directly
    # This avoids the OOM-then-fallback which leaves memory fragmented
    use_chunked = (pixels_per_frame > 768 * 512 and num_video_frames > 121) or \
                  (pixels_per_frame > 1024 * 768) or \
                  (total_pixels * 4 > free_mem * 0.3)  # Video tensor would use >30% of free mem

    # Use v2v-safe tiling context when in v2v mode to prevent edge pixel loss
    tiling_context = v2v_safe_tiling(vae) if v2v_mode else nullcontext()

    with tiling_context:
        if v2v_mode:
            print(f">>> V2V mode: using edge-safe tiling", flush=True)

        if use_chunked:
            print(f">>> High-res/long video detected, using chunked encoding directly...", flush=True)
            latents = _encode_video_chunked(pil_frames, num_video_frames, target_h, target_w,
                                             vae, device, vae_dtype, expected_latent_frames,
                                             use_deterministic=use_deterministic)
        else:
            # Try VAE's built-in encoding for smaller videos
            latents = _encode_video_full(pil_frames, num_video_frames, target_h, target_w,
                                          vae, device, vae_dtype, use_deterministic=use_deterministic,
                                          v2v_mode=v2v_mode)

    num_latent_frames = latents.shape[0]

    print(f">>> Video encoded: {num_video_frames} video frames -> {num_latent_frames} latent frames (expected: {expected_latent_frames})", flush=True)
    print(f">>> Latent shape: {latents.shape}", flush=True)

    return latents, scale_factor, num_video_frames


def _encode_video_full(pil_frames, num_video_frames, target_h, target_w, vae, device, vae_dtype, use_deterministic=True, v2v_mode=False):
    """
    Encode video using VAE's built-in temporal tiled encoding.
    Works well for lower resolutions or shorter videos.

    Uses musubi-tuner style deterministic encoding with .mode() when use_deterministic=True.

    When v2v_mode=True, uses opt_tiling=False to preserve the conservative tile sizes
    set by the v2v_safe_tiling context manager (prevents OOM from dynamic tile sizing).
    """
    import torch.nn.functional as F_torch

    with torch.no_grad():
        frame_tensors = []
        for i in range(num_video_frames):
            pil_image = pil_frames[i]
            image = F.pil_to_tensor(pil_image).unsqueeze(0).float()
            if image.shape[2] != target_h or image.shape[3] != target_w:
                image = F_torch.interpolate(image, size=(target_h, target_w), mode='bilinear', align_corners=False)
            image = image / 127.5 - 1.
            frame_tensors.append(image)

        # Stack frames: [N, C, H, W] -> [1, C, N, H, W]
        video_tensor = torch.cat(frame_tensors, dim=0)
        video_tensor = video_tensor.permute(1, 0, 2, 3).unsqueeze(0)
        video_tensor = video_tensor.to(device=device, dtype=vae_dtype)
        del frame_tensors

        # Use VAE's built-in tiled encoding (handles temporal chunking with proper blending)
        # In v2v mode, the v2v_safe_tiling context sets _skip_tile_setup flag to preserve
        # conservative tile sizes (so opt_tiling value doesn't matter in that case)
        print(f">>> Using VAE's built-in temporal tiled encoding with blending...", flush=True)
        latent_dist = vae.encode(video_tensor, opt_tiling=True).latent_dist
        # Use .mode() for deterministic or .sample() for stochastic (musubi-tuner uses mode)
        latents = latent_dist.mode() if use_deterministic else latent_dist.sample()
        latents = latents.squeeze(0).permute(1, 2, 3, 0)
        latents = latents * vae.config.scaling_factor
        latents = latents.cpu()

        del video_tensor
        torch.cuda.empty_cache()

    return latents


def _encode_video_chunked(pil_frames, num_video_frames, target_h, target_w, vae, device, vae_dtype, expected_latent_frames, use_deterministic=True):
    """
    Encode video in smaller chunks for high-resolution or long videos.
    Mirrors the VAE's _temporal_tiled_encode but loads frames on-demand to save memory.

    Uses musubi-tuner style techniques:
    - Deterministic .mode() sampling for consistent results
    - Memory-efficient on-demand frame loading
    - Proper temporal blending at chunk boundaries
    """
    import torch.nn.functional as F_torch
    from tqdm import tqdm

    # Mirror VAE's temporal tiling parameters
    tile_sample_min_num_frames = vae.tile_sample_min_num_frames  # Default: 16
    tile_sample_stride_num_frames = vae.tile_sample_stride_num_frames  # Default: 12

    # For high res, use smaller temporal chunks to reduce peak memory
    # IMPORTANT: stride must be >= 4 to produce at least 1 latent frame per stride
    pixels_per_frame = target_h * target_w
    if pixels_per_frame >= 1024 * 768:
        # High res (1024x768 or larger) - use smaller temporal tiles
        # 9 pixel frames → 3 latent frames, stride 4 → 1 latent frame
        tile_sample_min_num_frames = 8
        tile_sample_stride_num_frames = 4
    elif pixels_per_frame > 768 * 512:
        # Medium-high res - use smaller temporal tiles
        tile_sample_min_num_frames = 12
        tile_sample_stride_num_frames = 8

    # For high res, also force spatial tiling by temporarily setting smaller tile dimensions
    # This ensures VAE's _encode triggers tiled_encode instead of direct encoding
    force_spatial_tiling = pixels_per_frame >= 768 * 512
    if force_spatial_tiling:
        old_tile_h = vae.tile_sample_min_height
        old_tile_w = vae.tile_sample_min_width
        old_stride_h = vae.tile_sample_stride_height
        old_stride_w = vae.tile_sample_stride_width
        old_use_tiling = vae.use_tiling
        # Use 256x256 tiles with overlap (musubi-tuner style)
        vae.tile_sample_min_height = 256
        vae.tile_sample_min_width = 256
        vae.tile_sample_stride_height = 192
        vae.tile_sample_stride_width = 192
        vae.use_tiling = True  # Ensure tiling is enabled
        print(f">>> Forcing spatial tiling: 256x256 tiles for {target_w}x{target_h} resolution", flush=True)

    temporal_tile_frames = tile_sample_min_num_frames + 1  # +1 for overlap (same as VAE)
    temporal_stride_frames = tile_sample_stride_num_frames

    # Calculate blend region in latent space (same formula as VAE)
    tile_latent_min_num_frames = tile_sample_min_num_frames // 4
    tile_latent_stride_num_frames = tile_sample_stride_num_frames // 4
    blend_num_frames = tile_latent_min_num_frames - tile_latent_stride_num_frames

    # Calculate temporal chunks (same as VAE's _temporal_tiled_encode)
    temporal_chunks = list(range(
        0,
        num_video_frames - tile_sample_min_num_frames + 1,
        tile_sample_stride_num_frames,
    ))
    # Ensure we cover all frames
    if not temporal_chunks:
        temporal_chunks = [0]

    print(f">>> Chunked encoding: {len(temporal_chunks)} temporal chunks of {temporal_tile_frames} frames (stride: {temporal_stride_frames})", flush=True)

    latent_rows = []

    try:
        with torch.no_grad():
            for chunk_idx, start_frame in enumerate(tqdm(temporal_chunks, desc="VAE temporal encoding", unit="chunk")):
                # Same frame selection as VAE: i : i + tile_sample_min_num_frames + 1
                end_frame = min(start_frame + temporal_tile_frames, num_video_frames)

                # Load only this chunk's frames to GPU (memory efficient - musubi-tuner style)
                chunk_tensors = []
                for i in range(start_frame, end_frame):
                    pil_image = pil_frames[i]
                    image = F.pil_to_tensor(pil_image).unsqueeze(0).float()
                    if image.shape[2] != target_h or image.shape[3] != target_w:
                        image = F_torch.interpolate(image, size=(target_h, target_w), mode='bilinear', align_corners=False)
                    image = image / 127.5 - 1.
                    chunk_tensors.append(image)

                # Stack: [N, C, H, W] -> [1, C, N, H, W]
                chunk = torch.cat(chunk_tensors, dim=0)
                chunk = chunk.permute(1, 0, 2, 3).unsqueeze(0)
                chunk = chunk.to(device=device, dtype=vae_dtype)
                del chunk_tensors

                # Encode this chunk - VAE will apply spatial tiling based on our settings
                # Using opt_tiling=True so VAE computes optimal tiling based on our manually set tile dimensions
                latent_dist = vae.encode(chunk, opt_tiling=False).latent_dist
                # Use .mode() for deterministic or .sample() for stochastic (musubi-tuner uses mode)
                latent_chunk = latent_dist.mode() if use_deterministic else latent_dist.sample()
                latent_chunk = latent_chunk * vae.config.scaling_factor

                # Drop first latent frame for subsequent chunks (same as VAE: tile[:, :, 1:, :, :])
                if chunk_idx > 0:
                    latent_chunk = latent_chunk[:, :, 1:, :, :]

                # Keep in VAE's format [B, C, T, H, W] for blending, move to CPU
                latent_rows.append(latent_chunk.cpu())

                del chunk, latent_chunk
                torch.cuda.empty_cache()
    finally:
        # Restore VAE's original tile settings
        if force_spatial_tiling:
            vae.tile_sample_min_height = old_tile_h
            vae.tile_sample_min_width = old_tile_w
            vae.tile_sample_stride_height = old_stride_h
            vae.tile_sample_stride_width = old_stride_w
            vae.use_tiling = old_use_tiling

    # Blend temporal chunks using VAE's blend_t approach (same logic as _temporal_tiled_encode)
    result_rows = []
    for i, tile in enumerate(latent_rows):
        if i > 0:
            # Apply temporal blending (VAE's blend_t operates on dim 2 which is temporal)
            tile = _blend_t_vae_format(latent_rows[i - 1], tile, blend_num_frames)
            # Take stride portion (except for last chunk which takes all remaining)
            t_lim = tile_latent_min_num_frames if i == len(latent_rows) - 1 else tile_latent_stride_num_frames
            result_rows.append(tile[:, :, :t_lim, :, :])
        else:
            # First chunk: take stride + 1 frames (same as VAE)
            result_rows.append(tile[:, :, :tile_latent_stride_num_frames + 1, :, :])

    # Concatenate and convert to output format
    latents = torch.cat(result_rows, dim=2)  # Concat on temporal dim
    latents = latents[:, :, :expected_latent_frames, :, :]  # Trim to expected
    latents = latents.squeeze(0).permute(1, 2, 3, 0)  # [C, T, H, W] -> [T, H, W, C]

    return latents


def _blend_t_vae_format(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
    """
    Temporal blending in VAE format [B, C, T, H, W].
    Mirrors VAE's blend_t method exactly.
    """
    blend_extent = min(a.shape[2], b.shape[2], blend_extent)
    if blend_extent <= 0:
        return b

    b = b.clone()
    for x in range(blend_extent):
        # Same formula as VAE's blend_t
        b[:, :, x, :, :] = a[:, :, -blend_extent + x, :, :] * (1 - x / blend_extent) + b[:, :, x, :, :] * (x / blend_extent)

    return b


def decode_video_from_latents(
    latents: torch.Tensor,
    vae,
    device: torch.device,
    output_format: str = "tensor",
) -> torch.Tensor:
    """
    Decode latent tensor back to video using musubi-tuner style decoding.

    Uses VAE's built-in tiled decoding with proper temporal and spatial blending.

    Args:
        latents: Latent tensor [T, H, W, C] or [B, C, T, H, W]
        vae: VAE model
        device: Device to use
        output_format: "tensor" returns [B, C, T, H, W], "frames" returns list of PIL images

    Returns:
        Decoded video tensor or list of PIL frames
    """
    from tqdm import tqdm

    # Convert to VAE format [B, C, T, H, W] if needed
    if latents.dim() == 4:
        # [T, H, W, C] -> [B, C, T, H, W]
        latents = latents.permute(3, 0, 1, 2).unsqueeze(0)
    elif latents.dim() == 5 and latents.shape[-1] == 16:
        # [B, T, H, W, C] -> [B, C, T, H, W]
        latents = latents.permute(0, 4, 1, 2, 3)

    # Unscale latents
    latents = latents / vae.config.scaling_factor

    vae_dtype = next(vae.parameters()).dtype
    latents = latents.to(device=device, dtype=vae_dtype)

    # Use VAE's built-in tiled decoding (handles temporal and spatial chunking with blending)
    with torch.no_grad():
        print(f">>> Decoding latents to video (musubi-tuner style)...", flush=True)
        print(f">>> Latent shape: {latents.shape}", flush=True)

        # VAE.decode handles tiling automatically based on latent size
        decoded = vae.decode(latents).sample

        # Clamp and convert to uint8
        decoded = ((decoded.clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8)

    print(f">>> Decoded video shape: {decoded.shape}", flush=True)

    if output_format == "frames":
        # Convert to list of PIL images
        from PIL import Image
        frames = []
        for t in range(decoded.shape[2]):
            frame = decoded[0, :, t].permute(1, 2, 0).cpu().numpy()
            frames.append(Image.fromarray(frame))
        return frames

    return decoded


def spatial_tiled_decode(
    vae,
    z: torch.Tensor,
    tile_size: int = 256,
    tile_overlap: float = 0.25,
) -> torch.Tensor:
    """
    Decode latents using spatial tiling with smooth blending.
    From musubi-tuner: spatial_tiled_decode in hunyuan_video_1_5_vae.py

    Args:
        vae: VAE model
        z: Latent tensor [B, C, T, H, W]
        tile_size: Size of each tile in latent space
        tile_overlap: Overlap factor between tiles (0.0 to 0.5)

    Returns:
        Decoded video tensor [B, C, T, H, W]
    """
    B, C, T, H, W = z.shape
    spatial_compression = vae.spatial_compression_ratio  # Usually 8

    # Calculate tile parameters in latent space
    tile_size_latent = tile_size // spatial_compression
    overlap_size_latent = int(tile_size_latent * (1 - tile_overlap))
    blend_extent_pixels = int(tile_size * tile_overlap)

    # Output dimensions
    out_h = H * spatial_compression
    out_w = W * spatial_compression

    rows = []
    for i in range(0, H, overlap_size_latent):
        row = []
        for j in range(0, W, overlap_size_latent):
            # Extract tile
            tile = z[:, :, :, i:i + tile_size_latent, j:j + tile_size_latent]

            # Decode tile
            with torch.no_grad():
                tile = vae.post_quant_conv(tile)
                decoded_tile = vae.decoder(tile)

            row.append(decoded_tile)
        rows.append(row)

    # Blend tiles together
    tile_limit_pixels = tile_size - blend_extent_pixels
    result_rows = []
    for i, row in enumerate(rows):
        result_row = []
        for j, tile in enumerate(row):
            if i > 0:
                tile = blend_v(rows[i - 1][j], tile, blend_extent_pixels)
            if j > 0:
                tile = blend_h(row[j - 1], tile, blend_extent_pixels)
            result_row.append(tile[:, :, :, :tile_limit_pixels, :tile_limit_pixels])
        result_rows.append(torch.cat(result_row, dim=-1))

    decoded = torch.cat(result_rows, dim=-2)
    return decoded[:, :, :, :out_h, :out_w]


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


def get_first_frame_from_image(image, vae, device, alignment=16, use_deterministic=True):
    """
    Load and encode an image to latent space.

    Uses musubi-tuner style deterministic encoding with .mode() when use_deterministic=True.

    Args:
        image: Path to image or PIL Image
        vae: VAE model
        device: Device to use
        alignment: Pixel alignment for resizing (use 128 for NABLA attention)
        use_deterministic: If True, use .mode() for deterministic encoding (musubi-tuner style).
    """
    if isinstance(image, str):
        pil_image = Image.open(image).convert('RGB')
    elif isinstance(image, Image.Image):
        pil_image = image
    else:
        raise ValueError(f"unknown image type: {type(image)}")

    image = F.pil_to_tensor(pil_image).unsqueeze(0)
    image, k = resize_image(image, max_area=MAX_AREA, alignment=alignment)
    image = image / 127.5 - 1.

    with torch.no_grad():
        # Use the VAE's dtype to avoid dtype mismatch
        vae_dtype = next(vae.parameters()).dtype
        image = image.to(device=device, dtype=vae_dtype).transpose(0, 1).unsqueeze(0)
        latent_dist = vae.encode(image, opt_tiling=False).latent_dist
        # Use .mode() for deterministic or .sample() for stochastic (musubi-tuner uses mode)
        lat_image = latent_dist.mode() if use_deterministic else latent_dist.sample()
        lat_image = lat_image.squeeze(0).permute(1, 2, 3, 0) * vae.config.scaling_factor

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
        clip_prompt: str = None,
        save_path: str = None,
        progress: bool = True,
        preview: int = None,
        preview_suffix: str = None,
        stop_check=None,
        checkpoint_path=None,
        save_latents=None,
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
                print("\n" + "="*80)
                print("EXPANDED QWEN 2.5 PROMPT:")
                print("="*80)
                try:
                    print(caption)
                except UnicodeEncodeError:
                    # Windows console encoding (cp1252) can't handle some Unicode characters
                    print(caption.encode('utf-8', errors='replace').decode('utf-8', errors='replace'))
                print("="*80 + "\n")
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
            clip_prompt=clip_prompt,
            seed=seed,
            device=self.device_map["dit"],
            vae_device=self.device_map["vae"],
            progress=progress,
            offload=self.offload,
            force_offload=force_offload,
            previewer=previewer,
            preview_interval=preview,
            preview_suffix=preview_suffix,
            stop_check=stop_check,
            checkpoint_path=checkpoint_path,
            save_latents=save_latents,
        )

        # Handle checkpoint save (images will be None)
        if images is None:
            # Delete text encoder to free RAM
            del self.text_embedder
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            return None

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


class Kandinsky5DenoisePipeline:
    """
    Pipeline for video-to-video denoising (img2img style for videos).

    Takes an existing video, encodes it to latent space, adds controlled noise,
    denoises it with the DiT model, and decodes back to video. Useful for:
    - Smoothing artifacts at video join points
    - Light style transfer with low denoise strength
    - Reducing compression artifacts
    """

    def __init__(
        self,
        device_map: Union[str, torch.device, dict],
        dit,
        text_embedder,
        vae,
        local_dit_rank: int = 0,
        world_size: int = 1,
        conf=None,
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
        video_path: str,
        denoise_strength: float = 0.2,
        seed: int = None,
        num_steps: int = None,
        guidance_weight: float = None,
        scheduler_scale: float = 10.0,
        negative_caption: str = "",
        clip_prompt: str = None,
        save_path: str = None,
        progress: bool = True,
        chunk_seconds: float = 5.0,
        chunk_overlap: int = 4,
    ):
        """
        Denoise a video file.

        Args:
            text: Text prompt describing the video content
            video_path: Path to input video file
            denoise_strength: How much to denoise (0.1-0.5 typical). Higher = more change.
            seed: Random seed
            num_steps: Number of denoising steps (scaled by denoise_strength)
            guidance_weight: CFG weight (2-4 recommended for preservation)
            scheduler_scale: Scheduler scale
            negative_caption: Negative prompt
            clip_prompt: Optional separate CLIP prompt
            save_path: Path to save output video
            progress: Show progress bar
            chunk_seconds: Process video in chunks of this duration
            chunk_overlap: Number of frames to overlap between chunks

        Returns:
            Denoised video tensor
        """
        num_steps = self.num_steps if num_steps is None else num_steps
        guidance_weight = self.guidance_weight if guidance_weight is None else guidance_weight

        if seed is None:
            if self.local_dit_rank == 0:
                seed = torch.randint(2**32 - 1, (1,)).to(self.local_dit_rank)
            else:
                seed = torch.empty((1,), dtype=torch.int64).to(self.local_dit_rank)

            if self.world_size > 1:
                torch.distributed.broadcast(seed, 0)

            seed = seed.item()

        # Determine alignment based on attention type
        try:
            attention_type = self.conf.model.attention.type
        except (AttributeError, KeyError):
            attention_type = 'flash'
        alignment = 128 if attention_type == 'nabla' else 16

        force_offload = hasattr(self.dit, 'enable_block_swap') and self.dit.enable_block_swap

        # Load VAE for encoding
        log_vram_usage("BEFORE VAE ENCODE (DENOISE)", dit=self.dit, vae=self.vae, text_embedder=self.text_embedder)

        if self.offload or force_offload:
            self.vae = self.vae.to(self.device_map["vae"], non_blocking=True)

        # Calculate frames per chunk
        frames_per_chunk = int(chunk_seconds * 24 / 4 + 1)  # Match Kandinsky frame rate
        print(f">>> Processing video in chunks of ~{chunk_seconds}s ({frames_per_chunk} frames per chunk)", flush=True)

        # Encode full video to latents (v2v_mode ensures edge-safe tiling)
        video_latents, scale_factor, total_frames = encode_video_to_latents(
            video_path, self.vae, self.device_map["vae"], alignment=alignment, v2v_mode=True
        )

        # Offload VAE after encoding
        log_vram_usage("AFTER VAE ENCODE, BEFORE OFFLOAD (DENOISE)", dit=self.dit, vae=self.vae, text_embedder=self.text_embedder)

        if self.offload or force_offload:
            self.vae = self.vae.to("cpu", non_blocking=True)
            torch.cuda.empty_cache()

        # Process video in chunks
        all_outputs = []
        chunk_start = 0

        while chunk_start < total_frames:
            chunk_end = min(chunk_start + frames_per_chunk, total_frames)
            chunk_latents = video_latents[chunk_start:chunk_end]

            print(f"\n>>> Processing chunk: frames {chunk_start}-{chunk_end} ({chunk_latents.shape[0]} frames)", flush=True)

            # Reload text embedder for each chunk (it gets deleted after use)
            # For now, we only support single chunk processing
            if chunk_start > 0:
                print(">>> Warning: Multi-chunk processing requires reloading text embedder", flush=True)
                break

            # Load text embedder if needed
            if self.offload or force_offload:
                self.text_embedder = self.text_embedder.to(self.device_map["text_embedder"])

            # Use lower guidance weight for better preservation
            effective_guidance = min(guidance_weight, 4.0)

            denoised_images = generate_sample_denoise(
                video_latents=chunk_latents,
                caption=text,
                dit=self.dit,
                vae=self.vae,
                conf=self.conf,
                text_embedder=self.text_embedder,
                denoise_strength=denoise_strength,
                num_steps=num_steps,
                guidance_weight=effective_guidance,
                scheduler_scale=scheduler_scale,
                negative_caption=negative_caption,
                clip_prompt=clip_prompt,
                seed=seed,
                device=self.device_map["dit"],
                vae_device=self.device_map["vae"],
                progress=progress,
                offload=self.offload,
                force_offload=force_offload,
            )

            all_outputs.append(denoised_images)

            chunk_start = chunk_end - chunk_overlap
            if chunk_start >= total_frames - chunk_overlap:
                break

        # Concatenate all chunks
        if len(all_outputs) == 1:
            images = all_outputs[0]
        else:
            # TODO: Implement proper chunk blending for multi-chunk processing
            images = torch.cat([out for out in all_outputs], dim=2)

        # Handle rescaling if needed
        if scale_factor > 16:
            h, w = images.shape[-2:]
            images = F.resize(images[0], (int(h / scale_factor / 16), int(w / scale_factor / 16)))

        # Save output
        if self.local_dit_rank == 0 and save_path is not None:
            torchvision.io.write_video(
                save_path,
                images[0].float().permute(1, 2, 3, 0).cpu().numpy(),
                fps=24,
                options={"crf": "5"},
            )
            print(f">>> Saved denoised video to {save_path}", flush=True)

        return images
