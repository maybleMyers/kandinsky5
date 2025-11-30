import os
os.environ["TOKENIZERS_PARALLELISM"] = "False"

import math
import torch
from tqdm import tqdm

from .models.utils import fast_sta_nabla

# APG (Adaptive Projected Guidance) helpers for reducing color drift in long video generation
# Reference: https://arxiv.org/abs/2410.02416

class MomentumBuffer:
    """Momentum buffer for APG to smooth guidance updates using EMA."""
    def __init__(self, momentum: float = 0.9):
        # Momentum should be positive (0.9 = 90% old, 10% new)
        self.momentum = abs(momentum)  # Force positive
        self.running_average = None

    def update(self, update_value: torch.Tensor):
        if self.running_average is None:
            self.running_average = update_value.clone()
        else:
            # Proper EMA: new_avg = momentum * old_avg + (1 - momentum) * new_value
            self.running_average = (
                self.momentum * self.running_average +
                (1 - self.momentum) * update_value
            )
        return self.running_average


def project_per_frame(v0: torch.Tensor, v1: torch.Tensor):
    """
    Project v0 onto v1 per-frame to preserve temporal structure.

    Args:
        v0: Tensor of shape [batch, C, frames, H, W]
        v1: Tensor of shape [batch, C, frames, H, W]

    Returns:
        v0_parallel, v0_orthogonal - both same shape as input
    """
    dtype = v0.dtype
    v0, v1 = v0.double(), v1.double()

    # Normalize per-frame (over C, H, W only, keeping frames separate)
    # dims: [batch=0, C=1, frames=2, H=3, W=4]
    # Sum over C, H, W (dims 1, 3, 4) but keep frames (dim 2) separate
    v1_norm = torch.sqrt((v1 * v1).sum(dim=(1, 3, 4), keepdim=True) + 1e-8)
    v1_normalized = v1 / v1_norm

    # Compute projection coefficient per-frame
    # dot product over C, H, W dimensions only
    dot_product = (v0 * v1_normalized).sum(dim=(1, 3, 4), keepdim=True)

    # Parallel component (projection of v0 onto v1)
    v0_parallel = dot_product * v1_normalized

    # Orthogonal component (what's left after removing parallel)
    v0_orthogonal = v0 - v0_parallel

    return v0_parallel.to(dtype), v0_orthogonal.to(dtype)


def adaptive_projected_guidance(
    diff: torch.Tensor,
    pred_cond: torch.Tensor,
    momentum_buffer: MomentumBuffer = None,
    norm_threshold: float = 55.0,
    parallel_scale: float = 0.3,
):
    """
    Apply Adaptive Projected Guidance to reduce color drift while maintaining coherence.

    This works by decomposing the CFG difference into:
    - Parallel component: aligns with conditional prediction (content/structure)
    - Orthogonal component: perpendicular to conditional (style/color adjustments)

    We keep a blend of both to maintain coherence while reducing drift.

    Args:
        diff: Difference between conditional and unconditional predictions [B, C, T, H, W]
        pred_cond: Conditional prediction [B, C, T, H, W]
        momentum_buffer: Optional momentum buffer for temporal smoothing
        norm_threshold: Per-frame norm clipping threshold (0 to disable)
        parallel_scale: How much of parallel component to keep (0-1, default 0.3)
                       1.0 = normal CFG, 0.0 = only orthogonal

    Returns:
        Adjusted guidance direction (blend of parallel and orthogonal)
    """
    # Apply momentum smoothing if enabled
    if momentum_buffer is not None:
        diff = momentum_buffer.update(diff)

    # Apply per-frame norm clipping to prevent any single frame from dominating
    if norm_threshold > 0:
        # Compute norm per-frame (over C, H, W)
        diff_norm = torch.sqrt((diff * diff).sum(dim=(1, 3, 4), keepdim=True) + 1e-8)
        # Clip frames that exceed threshold
        scale_factor = torch.clamp(norm_threshold / diff_norm, max=1.0)
        diff = diff * scale_factor

    # Project per-frame to preserve temporal structure
    diff_parallel, diff_orthogonal = project_per_frame(diff, pred_cond)

    # Return blend: keep some parallel (for coherence) + orthogonal (for drift reduction)
    # parallel_scale=1.0 means normal CFG, parallel_scale=0.0 means only orthogonal
    return parallel_scale * diff_parallel + diff_orthogonal


def adaptive_mean_std_normalization(source, reference):
    source_mean = source.mean(dim=(1,2,3),keepdim=True)
    source_std = source.std(dim=(1,2,3),keepdim=True)
    #magic constants - limit changes in latents
    clump_mean_low = 0.05
    clump_mean_high = 0.1
    clump_std_low = 0.1
    clump_std_high = 0.25

    reference_mean = torch.clamp(reference.mean(), source_mean - clump_mean_low, source_mean + clump_mean_high)
    reference_std = torch.clamp(reference.std(), source_std - clump_std_low, source_std + clump_std_high)

    # normalization
    normalized = (source - source_mean) / source_std
    normalized = normalized * reference_std + reference_mean

    return normalized

def normalize_first_frame(latents, reference_frames=5, clump_values=False):
    latents_copy = latents.clone()
    samples = latents_copy

    if samples.shape[0] <= 1:
        return latents  # Only one frame, no normalization needed
    nFr = 4
    first_frames = samples[:nFr]
    reference_frames_data = samples[nFr:nFr+min(reference_frames, samples.shape[0]-1)]

    normalized_first = adaptive_mean_std_normalization(first_frames, reference_frames_data)
    if clump_values:
        min_val = reference_frames_data.min()
        max_val = reference_frames_data.max()
        normalized_first = torch.clamp(normalized_first, min_val, max_val)

    samples[:nFr] = normalized_first

    return samples


def normalize_generated_frames_to_conditioning(latents, num_cond_frames=4, transition_frames=5, clump_values=False):
    """
    Normalize generated frames to match the statistics of conditioning frames.
    This is the correct direction for video continuation - generated frames should
    match the style/appearance of the input conditioning frames.

    Args:
        latents: Full latent tensor [total_frames, H, W, C] with cond + generated
        num_cond_frames: Number of conditioning frames at the start
        transition_frames: Number of generated frames to normalize for smooth transition
        clump_values: Whether to clamp normalized values to reference range

    Returns:
        Normalized latents with smooth transition from conditioning to generated
    """
    latents_copy = latents.clone()
    samples = latents_copy

    if samples.shape[0] <= num_cond_frames:
        return latents  # No generated frames to normalize

    # Conditioning frames are the reference (clean input video)
    cond_frames = samples[:num_cond_frames]

    # Generated frames to normalize (first few after conditioning for smooth transition)
    num_gen_frames = min(transition_frames, samples.shape[0] - num_cond_frames)
    gen_frames = samples[num_cond_frames:num_cond_frames + num_gen_frames]

    # Normalize generated frames to match conditioning frame statistics
    normalized_gen = adaptive_mean_std_normalization(gen_frames, cond_frames)

    if clump_values:
        min_val = cond_frames.min()
        max_val = cond_frames.max()
        normalized_gen = torch.clamp(normalized_gen, min_val, max_val)

    samples[num_cond_frames:num_cond_frames + num_gen_frames] = normalized_gen

    return samples


def normalize_join_frames(latents, num_start_cond=1, num_end_cond=1, transition_frames=4, clump_values=True, gradual_blend=True):
    """
    Normalize generated frames for video joining/image interpolation.
    Handles dual-ended conditioning by normalizing frames near both the start and end.

    Args:
        latents: Full latent tensor [total_frames, H, W, C]
        num_start_cond: Number of conditioning frames at the start
        num_end_cond: Number of conditioning frames at the end
        transition_frames: Number of frames to normalize at each end
        clump_values: Whether to clamp normalized values to reference range
        gradual_blend: If True, normalize ALL frames with position-based blending

    Returns:
        Normalized latents with smooth transitions at both ends
    """
    latents_copy = latents.clone()
    samples = latents_copy
    total_frames = samples.shape[0]

    # Calculate generated frame range
    gen_start = num_start_cond
    gen_end = total_frames - num_end_cond

    if gen_start >= gen_end:
        return latents  # No generated frames to normalize

    start_cond_frames = samples[:num_start_cond]
    end_cond_frames = samples[-num_end_cond:] if num_end_cond > 0 else samples[-1:]

    if gradual_blend:
        # Normalize ALL generated frames with position-based weighting
        # First half uses start reference, second half transitions to end reference
        for frame_idx in range(gen_start, gen_end):
            progress = (frame_idx - gen_start) / max(1, (gen_end - gen_start - 1))  # 0 to 1
            frame = samples[frame_idx:frame_idx+1]  # Keep dimension

            # Normalize to both references
            norm_to_start = adaptive_mean_std_normalization(frame, start_cond_frames)
            norm_to_end = adaptive_mean_std_normalization(frame, end_cond_frames)

            # Blend based on position: first half -> start, second half -> end
            # Smooth transition using cosine
            blend_weight = 0.5 * (1 - math.cos(progress * math.pi))  # 0 at start, 1 at end
            normalized = (1 - blend_weight) * norm_to_start + blend_weight * norm_to_end

            if clump_values:
                # Blend the clamp ranges too
                min_val = (1 - blend_weight) * start_cond_frames.min() + blend_weight * end_cond_frames.min()
                max_val = (1 - blend_weight) * start_cond_frames.max() + blend_weight * end_cond_frames.max()
                normalized = torch.clamp(normalized, min_val, max_val)

            samples[frame_idx] = normalized[0]
    else:
        # Original edge-only normalization
        num_start_normalize = min(transition_frames, gen_end - gen_start)
        if num_start_normalize > 0:
            start_gen_frames = samples[gen_start:gen_start + num_start_normalize]
            normalized_start = adaptive_mean_std_normalization(start_gen_frames, start_cond_frames)
            if clump_values:
                min_val = start_cond_frames.min()
                max_val = start_cond_frames.max()
                normalized_start = torch.clamp(normalized_start, min_val, max_val)
            samples[gen_start:gen_start + num_start_normalize] = normalized_start

        num_end_normalize = min(transition_frames, gen_end - gen_start)
        if num_end_normalize > 0:
            end_gen_frames = samples[gen_end - num_end_normalize:gen_end]
            normalized_end = adaptive_mean_std_normalization(end_gen_frames, end_cond_frames)
            if clump_values:
                min_val = end_cond_frames.min()
                max_val = end_cond_frames.max()
                normalized_end = torch.clamp(normalized_end, min_val, max_val)
            samples[gen_end - num_end_normalize:gen_end] = normalized_end

    return samples


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


def get_sparse_params(conf, batch_embeds, device):
    assert conf.model.dit_params.patch_size[0] == 1
    T, H, W, _ = batch_embeds["visual"].shape
    T, H, W = (
        T // conf.model.dit_params.patch_size[0],
        H // conf.model.dit_params.patch_size[1],
        W // conf.model.dit_params.patch_size[2],
    )

    # Check if attention config exists and is NABLA type
    try:
        attention_type = conf.model.attention.type
    except (AttributeError, KeyError):
        attention_type = None

    if attention_type == "nabla":
        sta_mask = fast_sta_nabla(T, H // 8, W // 8, conf.model.attention.wT,
                                  conf.model.attention.wH, conf.model.attention.wW, device=device)
        sparse_params = {
            "sta_mask": sta_mask.unsqueeze_(0).unsqueeze_(0),
            "attention_type": conf.model.attention.type,
            "to_fractal": True,
            "P": conf.model.attention.P,
            "wT": conf.model.attention.wT,
            "wW": conf.model.attention.wW,
            "wH": conf.model.attention.wH,
            "add_sta": conf.model.attention.add_sta,
            "visual_shape": (T, H, W),
            "method": getattr(conf.model.attention, "method", "topcdf"),
        }
    else:
        sparse_params = None

    return sparse_params


@torch.no_grad()
def get_velocity(
    dit,
    x,
    t,
    text_embeds,
    null_text_embeds,
    visual_rope_pos,
    text_rope_pos,
    null_text_rope_pos,
    guidance_weight,
    conf,
    sparse_params=None,
    attention_mask=None,
    null_attention_mask=None,
):
    with torch._dynamo.utils.disable_cache_limit():
        pred_velocity = dit(
            x,
            text_embeds["text_embeds"],
            text_embeds["pooled_embed"],
            t * 1000,
            visual_rope_pos,
            text_rope_pos,
            scale_factor=conf.metrics.scale_factor,
            sparse_params=sparse_params,
            attention_mask=attention_mask,
        )
        if abs(guidance_weight - 1.0) > 1e-6:
            uncond_pred_velocity = dit(
                x,
                null_text_embeds["text_embeds"],
                null_text_embeds["pooled_embed"],
                t * 1000,
                visual_rope_pos,
                null_text_rope_pos,
                scale_factor=conf.metrics.scale_factor,
                sparse_params=sparse_params,
                attention_mask=null_attention_mask,
            )
            pred_velocity = uncond_pred_velocity + guidance_weight * (
                pred_velocity - uncond_pred_velocity
            )
    return pred_velocity


@torch.no_grad()
def generate(
    model,
    device,
    shape,
    num_steps,
    text_embeds,
    null_text_embeds,
    visual_rope_pos,
    text_rope_pos,
    null_text_rope_pos,
    guidance_weight,
    scheduler_scale,
    first_frames,
    conf,
    progress=False,
    seed=6554,
    attention_mask=None,
    null_attention_mask=None,
    previewer=None,
    preview_interval=None,
    preview_suffix=None,
    stop_check=None,
):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    img = torch.randn(*shape, device=device, generator=g)

    # Store original noise for early-stop decode
    original_noise = img.clone()

    sparse_params = get_sparse_params(conf, {"visual": img}, device)
    timesteps = torch.linspace(1, 0, num_steps + 1, device=device)
    timesteps = scheduler_scale * timesteps / (1 + (scheduler_scale - 1) * timesteps)

    for i, (timestep, timestep_diff) in enumerate(tqdm(list(zip(timesteps[:-1], torch.diff(timesteps))))):
        time = timestep.unsqueeze(0)
        if model.visual_cond:
            visual_cond = torch.zeros_like(img)
            visual_cond_mask = torch.zeros(
                [*img.shape[:-1], 1], dtype=img.dtype, device=img.device
            )
            if first_frames is not None:
                first_frames = first_frames.to(device=visual_cond.device, dtype=visual_cond.dtype)
                img[:1] = first_frames
                visual_cond_mask[:1] = 1
            model_input = torch.cat([img, visual_cond, visual_cond_mask], dim=-1)
        else:
            model_input = img
        pred_velocity = get_velocity(
            model,
            model_input,
            time,
            text_embeds,
            null_text_embeds,
            visual_rope_pos,
            text_rope_pos,
            null_text_rope_pos,
            guidance_weight,
            conf,
            sparse_params=sparse_params,
            attention_mask=attention_mask,
            null_attention_mask=null_attention_mask,
        )
        img = img + timestep_diff * pred_velocity

        # Check for early stop request
        if stop_check is not None:
            action = stop_check()
            if action in ("decode", "save"):
                print(f"\n>>> Early stop requested at step {i + 1}/{num_steps} - action: {action}", flush=True)
                return {
                    "action": action,
                    "latents": img,
                    "step": i + 1,
                    "total_steps": num_steps,
                    "original_noise": original_noise,
                    "timesteps": timesteps
                }

        if previewer is not None and preview_interval and (i + 1) % preview_interval == 0 and (i + 1) < num_steps:
            import sys
            print(f"\n>>> PREVIEW TRIGGER at step {i + 1}/{num_steps} (interval={preview_interval})", flush=True)
            sys.stdout.flush()
            print(f">>> img shape before permute: {img.shape}", flush=True)
            try:
                preview_latent = img.permute(3, 0, 1, 2).unsqueeze(0)
                print(f">>> preview_latent shape after permute+unsqueeze: {preview_latent.shape}", flush=True)
                previewer.preview(preview_latent.squeeze(0), i, preview_suffix=preview_suffix)
                print(f">>> Preview completed successfully", flush=True)
                sys.stdout.flush()
            except Exception as e:
                print(f">>> ERROR during preview generation at step {i + 1}: {e}", flush=True)
                import traceback
                traceback.print_exc()
                sys.stdout.flush()
    return img


def generate_sample(
    shape,
    caption,
    dit,
    vae,
    conf,
    text_embedder,
    num_steps=25,
    guidance_weight=5.0,
    scheduler_scale=1,
    negative_caption="",
    clip_prompt=None,
    seed=6554,
    device="cuda",
    vae_device="cuda",
    text_embedder_device="cuda",
    progress=True,
    offload=False,
    force_offload=False,
    image_vae=False,
    previewer=None,
    preview_interval=None,
    preview_suffix=None,
    stop_check=None,
    checkpoint_path=None,
    save_latents=None,
):
    bs, duration, height, width, dim = shape
    if duration == 1:
        type_of_content = "image"
    else:
        type_of_content = "video"

    with torch.no_grad():
        # Pass clip_texts if a separate clip_prompt is provided
        clip_texts = [clip_prompt] if clip_prompt else None
        bs_text_embed, text_cu_seqlens, attention_mask = text_embedder.encode(
            [caption], type_of_content=type_of_content, clip_texts=clip_texts
        )
        bs_null_text_embed, null_text_cu_seqlens, null_attention_mask = text_embedder.encode(
            [negative_caption], type_of_content=type_of_content
        )

    # Clean up text embedder after encoding to free VRAM and RAM
    # Text embedder is no longer needed after this point
    if offload or force_offload:
        text_embedder = text_embedder.to('cpu')
    # Delete text embedder components to free memory
    del text_embedder.embedder.model
    del text_embedder.clip_embedder.model
    del text_embedder
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    for key in bs_text_embed:
        bs_text_embed[key] = bs_text_embed[key].to(device=device,dtype=torch.bfloat16)
        bs_null_text_embed[key] = bs_null_text_embed[key].to(device=device,dtype=torch.bfloat16)
    text_cu_seqlens = text_cu_seqlens.to(device=device)[-1].item()
    null_text_cu_seqlens = null_text_cu_seqlens.to(device=device)[-1].item()
    attention_mask = attention_mask.to(device=device)
    null_attention_mask = null_attention_mask.to(device=device)

    visual_rope_pos = [
        torch.arange(duration),
        torch.arange(shape[-3] // conf.model.dit_params.patch_size[1]),
        torch.arange(shape[-2] // conf.model.dit_params.patch_size[2]),
    ]
    text_rope_pos = torch.arange(text_cu_seqlens)
    null_text_rope_pos = torch.arange(null_text_cu_seqlens)

    # Log VRAM before DiT inference
    log_vram_usage("BEFORE DiT INFERENCE (T2V)", dit=dit, vae=vae, text_embedder=None)

    if offload or force_offload:
        dit.to(device, non_blocking=True)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            result = generate(
                dit,
                device,
                (bs * duration, height, width, dim),
                num_steps,
                bs_text_embed,
                bs_null_text_embed,
                visual_rope_pos,
                text_rope_pos,
                null_text_rope_pos,
                guidance_weight,
                scheduler_scale,
                None,
                conf,
                seed=seed,
                progress=progress,
                attention_mask=attention_mask,
                null_attention_mask=null_attention_mask,
                previewer=previewer,
                preview_interval=preview_interval,
                preview_suffix=preview_suffix,
                stop_check=stop_check,
            )

    # Handle early stop results
    if isinstance(result, dict) and result.get("action"):
        action = result["action"]
        latent_visual = result["latents"]
        step = result["step"]
        total_steps = result["total_steps"]
        original_noise = result.get("original_noise")
        timesteps = result.get("timesteps")

        if action == "save" and checkpoint_path:
            # Save checkpoint for later resumption
            checkpoint = {
                "latents": latent_visual.cpu(),
                "step": step,
                "total_steps": total_steps,
                "seed": seed,
                "text_embeds": {k: v.cpu() for k, v in bs_text_embed.items()},
                "null_text_embeds": {k: v.cpu() for k, v in bs_null_text_embed.items()},
                "visual_rope_pos": [v.cpu() for v in visual_rope_pos],
                "text_rope_pos": text_rope_pos.cpu() if hasattr(text_rope_pos, 'cpu') else text_rope_pos,
                "null_text_rope_pos": null_text_rope_pos.cpu() if hasattr(null_text_rope_pos, 'cpu') else null_text_rope_pos,
                "shape": shape,
                "guidance_weight": guidance_weight,
                "scheduler_scale": scheduler_scale,
            }
            torch.save(checkpoint, checkpoint_path)
            print(f">>> Checkpoint saved to {checkpoint_path} at step {step}/{total_steps}", flush=True)
            return None  # Signal that we saved instead of decoding

        # For "decode" action, subtract remaining noise before decoding
        print(f">>> Decoding video from step {step}/{total_steps}", flush=True)

        # Subtract remaining noise (like preview does) to get clean latents
        if original_noise is not None and timesteps is not None:
            # timesteps[step] gives the noise level at the current step
            # We need to subtract this portion of the original noise
            noise_remaining = timesteps[step]
            latent_visual = latent_visual - (original_noise * noise_remaining)
            print(f">>> Subtracted {noise_remaining.item():.4f} of original noise", flush=True)
    else:
        latent_visual = result

    # Save latents before VAE decoding if requested
    if save_latents:
        latent_checkpoint = {
            "latents": latent_visual.cpu(),
            "shape": shape,
            "mode": "t2i" if duration == 1 else "t2v",
            "vae_scaling_factor": vae.config.scaling_factor,
            "latents_dtype": str(latent_visual.dtype),
        }
        torch.save(latent_checkpoint, save_latents)
        print(f">>> Latents saved to {save_latents}", flush=True)

    # Offload DiT before VAE decode to free up VRAM
    # For block swapping, explicitly offload all blocks first
    if hasattr(dit, 'offload_all_blocks'):
        dit.offload_all_blocks()

    if offload or force_offload:
        dit = dit.to('cpu', non_blocking=True)
    torch.cuda.empty_cache()

    # Log VRAM after DiT offload, before VAE decode
    log_vram_usage("AFTER DiT OFFLOAD, BEFORE VAE DECODE (T2V)", dit=dit, vae=vae, text_embedder=None)

    if offload or force_offload:
        vae = vae.to(vae_device, non_blocking=True)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            images = latent_visual.reshape(
                bs,
                -1,
                latent_visual.shape[-3],
                latent_visual.shape[-2],
                latent_visual.shape[-1],
            )
            images = images.to(device=vae_device)
            images = (images / vae.config.scaling_factor).permute(0, 4, 1, 2, 3)
            if image_vae:
                images = images[:,:,0]
            images = vae.decode(images).sample
            images = ((images.clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8)

    # Log VRAM after VAE decode, before VAE offload
    log_vram_usage("AFTER VAE DECODE, BEFORE OFFLOAD (T2V)", dit=dit, vae=vae, text_embedder=None)

    # Offload VAE after decode to free VRAM
    if offload or force_offload:
        vae = vae.to('cpu', non_blocking=True)
    torch.cuda.empty_cache()

    return images


@torch.no_grad()
def compute_kv_cache(
    model,
    device,
    cond_latents,
    text_embeds,
    text_rope_pos,
    conf,
    attention_mask=None,
):
    """
    Pre-compute KV cache for conditioning frames.

    Args:
        model: DiT model
        device: Device to use
        cond_latents: Conditioning latents [num_cond_frames, H, W, C]
        text_embeds: Text embeddings
        text_rope_pos: Text RoPE positions
        conf: Model configuration
        attention_mask: Attention mask

    Returns:
        KV cache dictionary {block_idx: (K, V)}
    """
    # Create timestep=0 for conditioning frames (they are "clean")
    num_cond_frames = cond_latents.shape[0]
    height, width = cond_latents.shape[1:3]
    timestep = torch.zeros(1, device=device)

    # Build model input for conditioning frames
    if model.visual_cond:
        visual_cond = torch.zeros_like(cond_latents)
        visual_cond_mask = torch.ones(
            [*cond_latents.shape[:-1], 1], dtype=cond_latents.dtype, device=cond_latents.device
        )
        model_input = torch.cat([cond_latents, visual_cond, visual_cond_mask], dim=-1)
    else:
        model_input = cond_latents

    # Visual RoPE positions for conditioning frames ONLY
    visual_rope_pos_cond = [
        torch.arange(num_cond_frames),
        torch.arange(height // conf.model.dit_params.patch_size[1]),
        torch.arange(width // conf.model.dit_params.patch_size[2]),
    ]

    sparse_params = get_sparse_params(conf, {"visual": cond_latents}, device)

    # Forward pass with return_kv=True
    with torch._dynamo.utils.disable_cache_limit():
        _, kv_cache_dict = model(
            model_input,
            text_embeds["text_embeds"],
            text_embeds["pooled_embed"],
            timestep * 1000,
            visual_rope_pos_cond,
            text_rope_pos,
            scale_factor=conf.metrics.scale_factor,
            sparse_params=sparse_params,
            attention_mask=attention_mask,
            return_kv=True,
        )

    return kv_cache_dict


@torch.no_grad()
def generate_v2v(
    model,
    device,
    shape,
    num_steps,
    text_embeds,
    null_text_embeds,
    visual_rope_pos,
    text_rope_pos,
    null_text_rope_pos,
    guidance_weight,
    scheduler_scale,
    cond_latents,
    conf,
    progress=False,
    seed=6554,
    attention_mask=None,
    null_attention_mask=None,
    previewer=None,
    preview_interval=None,
    preview_suffix=None,
    stop_check=None,
    use_apg=False,
    apg_momentum=0.9,
    apg_norm_threshold=55.0,
    apg_parallel_scale=0.3,
):
    """
    Generate video continuation using visual conditioning (like I2V but with multiple frames).

    Args:
        model: DiT model
        device: Device to use
        shape: Shape of FULL output (bs * total_frames, height, width, dim)
        num_steps: Number of denoising steps
        text_embeds: Text embeddings
        null_text_embeds: Null text embeddings for CFG
        visual_rope_pos: Visual RoPE positions for full sequence
        text_rope_pos: Text RoPE positions
        null_text_rope_pos: Null text RoPE positions
        guidance_weight: CFG weight
        scheduler_scale: Scheduler scale
        cond_latents: Conditioning latents [num_cond_frames, H, W, C]
        conf: Model configuration
        ...

    Returns:
        Generated latents for full sequence
    """
    num_cond_frames = cond_latents.shape[0]

    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    # Generate noise for FULL sequence (cond + new frames)
    img = torch.randn(*shape, device=device, generator=g)

    # Store original noise for early-stop decode
    original_noise = img.clone()

    # Store noise for conditioning frames - will be used to add appropriate noise at each step
    cond_noise = img[:num_cond_frames].clone()

    sparse_params = get_sparse_params(conf, {"visual": img}, device)
    timesteps = torch.linspace(1, 0, num_steps + 1, device=device)
    timesteps = scheduler_scale * timesteps / (1 + (scheduler_scale - 1) * timesteps)

    # Initialize APG momentum buffer if enabled
    momentum_buffer = MomentumBuffer(apg_momentum) if use_apg else None

    for i, (timestep, timestep_diff) in enumerate(tqdm(list(zip(timesteps[:-1], torch.diff(timesteps))))):
        time = timestep.unsqueeze(0)

        if model.visual_cond:
            visual_cond = torch.zeros_like(img)
            visual_cond_mask = torch.zeros(
                [*img.shape[:-1], 1], dtype=img.dtype, device=img.device
            )
            cond_latents_device = cond_latents.to(device=img.device, dtype=img.dtype)
            img[:num_cond_frames] = cond_latents_device
            visual_cond_mask[:num_cond_frames] = 1

            model_input = torch.cat([img, visual_cond, visual_cond_mask], dim=-1)
        else:
            model_input = img

        # Standard forward pass (no KV cache needed)
        with torch._dynamo.utils.disable_cache_limit():
            pred_velocity = model(
                model_input,
                text_embeds["text_embeds"],
                text_embeds["pooled_embed"],
                time * 1000,
                visual_rope_pos,
                text_rope_pos,
                scale_factor=conf.metrics.scale_factor,
                sparse_params=sparse_params,
                attention_mask=attention_mask,
            )

            # CFG
            if abs(guidance_weight - 1.0) > 1e-6:
                uncond_pred_velocity = model(
                    model_input,
                    null_text_embeds["text_embeds"],
                    null_text_embeds["pooled_embed"],
                    time * 1000,
                    visual_rope_pos,
                    null_text_rope_pos,
                    scale_factor=conf.metrics.scale_factor,
                    sparse_params=sparse_params,
                    attention_mask=null_attention_mask,
                )

                if use_apg:
                    # Adaptive Projected Guidance to reduce color drift
                    diff = pred_velocity - uncond_pred_velocity
                    # Permute for APG: [frames, H, W, C] -> [1, C, frames, H, W]
                    diff_reshaped = diff.permute(3, 0, 1, 2).unsqueeze(0)
                    pred_reshaped = pred_velocity.permute(3, 0, 1, 2).unsqueeze(0)

                    apg_result = adaptive_projected_guidance(
                        diff_reshaped, pred_reshaped, momentum_buffer,
                        apg_norm_threshold, apg_parallel_scale
                    )
                    # Permute back: [1, C, frames, H, W] -> [frames, H, W, C]
                    apg_result = apg_result.squeeze(0).permute(1, 2, 3, 0)
                    pred_velocity = uncond_pred_velocity + guidance_weight * apg_result
                else:
                    pred_velocity = uncond_pred_velocity + guidance_weight * (
                        pred_velocity - uncond_pred_velocity
                    )

        img = img + timestep_diff * pred_velocity

        # Check for early stop request
        if stop_check is not None:
            action = stop_check()
            if action in ("decode", "save"):
                print(f"\n>>> Early stop requested at step {i + 1}/{num_steps} - action: {action}", flush=True)
                return {
                    "action": action,
                    "latents": img,
                    "step": i + 1,
                    "total_steps": num_steps,
                    "original_noise": original_noise,
                    "timesteps": timesteps
                }

        if previewer is not None and preview_interval and (i + 1) % preview_interval == 0 and (i + 1) < num_steps:
            import sys
            print(f"\n>>> PREVIEW TRIGGER at step {i + 1}/{num_steps} (interval={preview_interval})", flush=True)
            sys.stdout.flush()
            try:
                preview_latent = img[num_cond_frames:].permute(3, 0, 1, 2).unsqueeze(0)
                previewer.preview(preview_latent.squeeze(0), i, preview_suffix=preview_suffix)
            except Exception as e:
                print(f">>> ERROR during preview generation at step {i + 1}: {e}", flush=True)

    # Ensure conditioning frames are exactly preserved in final output
    img[:num_cond_frames] = cond_latents.to(device=img.device, dtype=img.dtype)
    return img


def generate_v2v_join(
    model,
    device,
    shape,
    num_steps,
    text_embeds,
    null_text_embeds,
    visual_rope_pos,
    text_rope_pos,
    null_text_rope_pos,
    guidance_weight,
    scheduler_scale,
    start_cond_latents,
    end_cond_latents,
    conf,
    progress=False,
    seed=6554,
    attention_mask=None,
    null_attention_mask=None,
    previewer=None,
    preview_interval=None,
    preview_suffix=None,
    stop_check=None,
    use_apg=False,
    apg_momentum=0.9,
    apg_norm_threshold=55.0,
    apg_parallel_scale=0.3,
):
    """
    Generate video joining with dual conditioning (start and end frames).
    Creates a seamless transition between two videos by conditioning on:
    - Last N frames of video1 (start)
    - First N frames of video2 (end)

    Args:
        model: DiT model
        device: Device to use
        shape: Shape of FULL output (bs * total_frames, height, width, dim)
        num_steps: Number of denoising steps
        text_embeds: Text embeddings
        null_text_embeds: Null text embeddings for CFG
        visual_rope_pos: Visual RoPE positions for full sequence
        text_rope_pos: Text RoPE positions
        null_text_rope_pos: Null text RoPE positions
        guidance_weight: CFG weight
        scheduler_scale: Scheduler scale
        start_cond_latents: Start conditioning latents [num_cond_frames, H, W, C]
        end_cond_latents: End conditioning latents [num_cond_frames, H, W, C]
        conf: Model configuration
        ...

    Returns:
        Generated latents for full sequence with dual conditioning
    """
    num_start_cond_frames = start_cond_latents.shape[0]
    num_end_cond_frames = end_cond_latents.shape[0]

    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    # Generate noise for FULL sequence (start_cond + middle + end_cond)
    img = torch.randn(*shape, device=device, generator=g)

    # Initialize middle frames with position-dependent blend of start/end conditioning
    # This creates a gradual transition: first half uses start image as noise basis,
    # second half transitions to end image as noise basis
    total_frames = img.shape[0]
    start_cond_device = start_cond_latents.to(device=img.device, dtype=img.dtype)
    end_cond_device = end_cond_latents.to(device=img.device, dtype=img.dtype)

    for frame_idx in range(num_start_cond_frames, total_frames - num_end_cond_frames):
        # Calculate normalized position in the generated (middle) section
        gen_start = num_start_cond_frames
        gen_end = total_frames - num_end_cond_frames
        progress = (frame_idx - gen_start) / max(1, (gen_end - gen_start - 1))  # 0 to 1

        # Noise amount follows a bell curve: maximum in the middle, minimum at edges
        # Use cosine for smooth transition: cos(progress * pi) goes from 1 to -1
        noise_strength = 0.5 + 0.5 * abs(math.cos(progress * math.pi))  # Peaks at center

        # Blend between start and end conditioning based on position
        if progress < 0.5:
            # First half: primarily use start conditioning
            blend = progress * 2  # 0 to 1 over first half
            base_latent = (1 - blend) * start_cond_device[-1] + blend * (
                0.5 * start_cond_device[-1] + 0.5 * end_cond_device[0]
            )
        else:
            # Second half: transition to end conditioning
            blend = (progress - 0.5) * 2  # 0 to 1 over second half
            base_latent = (1 - blend) * (
                0.5 * start_cond_device[-1] + 0.5 * end_cond_device[0]
            ) + blend * end_cond_device[0]

        # Blend between base latent and noise based on noise_strength
        img[frame_idx] = (1 - noise_strength) * base_latent + noise_strength * img[frame_idx]

    # Store original noise for early-stop decode
    original_noise = img.clone()

    sparse_params = get_sparse_params(conf, {"visual": img}, device)
    timesteps = torch.linspace(1, 0, num_steps + 1, device=device)
    timesteps = scheduler_scale * timesteps / (1 + (scheduler_scale - 1) * timesteps)

    # Initialize APG momentum buffer if enabled
    momentum_buffer = MomentumBuffer(apg_momentum) if use_apg else None

    for i, (timestep, timestep_diff) in enumerate(tqdm(list(zip(timesteps[:-1], torch.diff(timesteps))))):
        time = timestep.unsqueeze(0)

        if model.visual_cond:
            visual_cond = torch.zeros_like(img)
            visual_cond_mask = torch.zeros(
                [*img.shape[:-1], 1], dtype=img.dtype, device=img.device
            )
            # Set BOTH start and end conditioning frames
            start_cond_device = start_cond_latents.to(device=img.device, dtype=img.dtype)
            end_cond_device = end_cond_latents.to(device=img.device, dtype=img.dtype)

            img[:num_start_cond_frames] = start_cond_device
            img[-num_end_cond_frames:] = end_cond_device
            visual_cond_mask[:num_start_cond_frames] = 1
            visual_cond_mask[-num_end_cond_frames:] = 1

            model_input = torch.cat([img, visual_cond, visual_cond_mask], dim=-1)
        else:
            model_input = img

        # Standard forward pass (no KV cache needed)
        with torch._dynamo.utils.disable_cache_limit():
            pred_velocity = model(
                model_input,
                text_embeds["text_embeds"],
                text_embeds["pooled_embed"],
                time * 1000,
                visual_rope_pos,
                text_rope_pos,
                scale_factor=conf.metrics.scale_factor,
                sparse_params=sparse_params,
                attention_mask=attention_mask,
            )

            # CFG
            if abs(guidance_weight - 1.0) > 1e-6:
                uncond_pred_velocity = model(
                    model_input,
                    null_text_embeds["text_embeds"],
                    null_text_embeds["pooled_embed"],
                    time * 1000,
                    visual_rope_pos,
                    null_text_rope_pos,
                    scale_factor=conf.metrics.scale_factor,
                    sparse_params=sparse_params,
                    attention_mask=null_attention_mask,
                )

                if use_apg:
                    # Adaptive Projected Guidance to reduce color drift
                    diff = pred_velocity - uncond_pred_velocity
                    # Permute for APG: [frames, H, W, C] -> [1, C, frames, H, W]
                    diff_reshaped = diff.permute(3, 0, 1, 2).unsqueeze(0)
                    pred_reshaped = pred_velocity.permute(3, 0, 1, 2).unsqueeze(0)

                    apg_result = adaptive_projected_guidance(
                        diff_reshaped, pred_reshaped, momentum_buffer,
                        apg_norm_threshold, apg_parallel_scale
                    )
                    # Permute back: [1, C, frames, H, W] -> [frames, H, W, C]
                    apg_result = apg_result.squeeze(0).permute(1, 2, 3, 0)
                    pred_velocity = uncond_pred_velocity + guidance_weight * apg_result
                else:
                    pred_velocity = uncond_pred_velocity + guidance_weight * (
                        pred_velocity - uncond_pred_velocity
                    )

        img = img + timestep_diff * pred_velocity

        # Check for early stop request
        if stop_check is not None:
            action = stop_check()
            if action in ("decode", "save"):
                print(f"\n>>> Early stop requested at step {i + 1}/{num_steps} - action: {action}", flush=True)
                return {
                    "action": action,
                    "latents": img,
                    "step": i + 1,
                    "total_steps": num_steps,
                    "original_noise": original_noise,
                    "timesteps": timesteps
                }

        if previewer is not None and preview_interval and (i + 1) % preview_interval == 0 and (i + 1) < num_steps:
            import sys
            print(f"\n>>> PREVIEW TRIGGER at step {i + 1}/{num_steps} (interval={preview_interval})", flush=True)
            sys.stdout.flush()
            try:
                # Preview only the middle section (exclude conditioning frames)
                preview_latent = img[num_start_cond_frames:-num_end_cond_frames].permute(3, 0, 1, 2).unsqueeze(0)
                previewer.preview(preview_latent.squeeze(0), i, preview_suffix=preview_suffix)
            except Exception as e:
                print(f">>> ERROR during preview generation at step {i + 1}: {e}", flush=True)

    # Ensure conditioning frames are exactly preserved in final output
    img[:num_start_cond_frames] = start_cond_latents.to(device=img.device, dtype=img.dtype)
    img[-num_end_cond_frames:] = end_cond_latents.to(device=img.device, dtype=img.dtype)
    return img


def generate_sample_v2v_join(
    shape,
    caption,
    dit,
    vae,
    conf,
    text_embedder,
    start_cond_latents,
    end_cond_latents,
    num_steps=50,
    guidance_weight=5.0,
    scheduler_scale=1,
    negative_caption="",
    clip_prompt=None,
    seed=6554,
    device="cuda",
    vae_device="cuda",
    progress=True,
    offload=False,
    force_offload=False,
    previewer=None,
    preview_interval=None,
    preview_suffix=None,
    stop_check=None,
    checkpoint_path=None,
    save_latents=None,
    use_apg=False,
    apg_momentum=0.9,
    apg_norm_threshold=55.0,
    apg_parallel_scale=0.3,
    normalize_latents=True,
    normalize_transition_frames=4,
):
    """
    Generate video joining with dual conditioning (start and end frames).
    Creates a seamless transition between two videos.

    Args:
        shape: Output shape (bs, duration, height, width, dim) - duration includes ALL frames
        caption: Text prompt for the transition
        dit: DiT model
        vae: VAE model
        conf: Configuration
        text_embedder: Text embedder
        start_cond_latents: Start conditioning latents [num_cond_frames, H, W, C] from video1
        end_cond_latents: End conditioning latents [num_cond_frames, H, W, C] from video2
        ...

    Returns:
        Generated video tensor with start, middle, and end sections
    """
    text_embedder.embedder.mode = "i2v"

    bs, total_frames, height, width, dim = shape
    num_start_cond_frames = start_cond_latents.shape[0]
    num_end_cond_frames = end_cond_latents.shape[0]

    type_of_content = "video"

    with torch.no_grad():
        clip_texts = [clip_prompt] if clip_prompt else None
        bs_text_embed, text_cu_seqlens, attention_mask = text_embedder.encode(
            [caption], type_of_content=type_of_content, clip_texts=clip_texts
        )
        bs_null_text_embed, null_text_cu_seqlens, null_attention_mask = text_embedder.encode(
            [negative_caption], type_of_content=type_of_content
        )

    # Clean up text embedder
    if offload or force_offload:
        text_embedder = text_embedder.to('cpu')
    del text_embedder.embedder.model
    del text_embedder.clip_embedder.model
    del text_embedder
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    for key in bs_text_embed:
        bs_text_embed[key] = bs_text_embed[key].to(device=device)
        bs_null_text_embed[key] = bs_null_text_embed[key].to(device=device)
    text_cu_seqlens = text_cu_seqlens.to(device=device)[-1].item()
    null_text_cu_seqlens = null_text_cu_seqlens.to(device=device)[-1].item()
    attention_mask = attention_mask.to(device=device)
    null_attention_mask = null_attention_mask.to(device=device)

    # Visual RoPE positions for FULL sequence (start_cond + middle + end_cond)
    visual_rope_pos = [
        torch.arange(total_frames),
        torch.arange(height // conf.model.dit_params.patch_size[1]),
        torch.arange(width // conf.model.dit_params.patch_size[2]),
    ]
    text_rope_pos = torch.arange(text_cu_seqlens)
    null_text_rope_pos = torch.arange(null_text_cu_seqlens)

    log_vram_usage("BEFORE DiT INFERENCE (V2V JOIN)", dit=dit, vae=vae, text_embedder=None)

    if offload or force_offload:
        dit.to(device, non_blocking=True)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            print(f">>> Generating video join with {num_start_cond_frames} start frames and {num_end_cond_frames} end frames...", flush=True)
            result = generate_v2v_join(
                dit,
                device,
                (bs * total_frames, height, width, dim),
                num_steps,
                bs_text_embed,
                bs_null_text_embed,
                visual_rope_pos,
                text_rope_pos,
                null_text_rope_pos,
                guidance_weight,
                scheduler_scale,
                start_cond_latents.to(device=device, dtype=torch.bfloat16),
                end_cond_latents.to(device=device, dtype=torch.bfloat16),
                conf,
                seed=seed,
                progress=progress,
                attention_mask=attention_mask,
                null_attention_mask=null_attention_mask,
                previewer=previewer,
                preview_interval=preview_interval,
                preview_suffix=preview_suffix,
                stop_check=stop_check,
                use_apg=use_apg,
                apg_momentum=apg_momentum,
                apg_norm_threshold=apg_norm_threshold,
                apg_parallel_scale=apg_parallel_scale,
            )

    # Handle early stop results
    if isinstance(result, dict) and result.get("action"):
        action = result["action"]
        latent_visual = result["latents"]
        step = result["step"]
        total_steps = result["total_steps"]
        original_noise = result.get("original_noise")
        timesteps = result.get("timesteps")

        if action == "save" and checkpoint_path:
            checkpoint = {
                "latents": latent_visual.cpu(),
                "step": step,
                "total_steps": total_steps,
                "seed": seed,
                "text_embeds": {k: v.cpu() for k, v in bs_text_embed.items()},
                "null_text_embeds": {k: v.cpu() for k, v in bs_null_text_embed.items()},
                "visual_rope_pos": [v.cpu() for v in visual_rope_pos],
                "text_rope_pos": text_rope_pos.cpu() if hasattr(text_rope_pos, 'cpu') else text_rope_pos,
                "null_text_rope_pos": null_text_rope_pos.cpu() if hasattr(null_text_rope_pos, 'cpu') else null_text_rope_pos,
                "start_cond_latents": start_cond_latents.cpu(),
                "end_cond_latents": end_cond_latents.cpu(),
                "shape": shape,
                "guidance_weight": guidance_weight,
                "scheduler_scale": scheduler_scale,
                "mode": "v2v_join",
            }
            torch.save(checkpoint, checkpoint_path)
            print(f">>> Checkpoint saved to {checkpoint_path} at step {step}/{total_steps}", flush=True)
            return None

        print(f">>> Decoding video from step {step}/{total_steps}", flush=True)

        if original_noise is not None and timesteps is not None:
            noise_remaining = timesteps[step]
            latent_visual = latent_visual - (original_noise * noise_remaining)
            print(f">>> Subtracted {noise_remaining.item():.4f} of original noise", flush=True)
    else:
        latent_visual = result

    # Apply normalization to smooth transitions at both ends
    if normalize_latents and normalize_transition_frames > 0:
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Ensure conditioning frames are preserved
                start_cond_latents_device = start_cond_latents.to(device=latent_visual.device, dtype=latent_visual.dtype)
                end_cond_latents_device = end_cond_latents.to(device=latent_visual.device, dtype=latent_visual.dtype)
                latent_visual[:num_start_cond_frames] = start_cond_latents_device
                latent_visual[-num_end_cond_frames:] = end_cond_latents_device
                # Normalize generated frames to match conditioning frames
                latent_visual = normalize_join_frames(
                    latent_visual,
                    num_start_cond=num_start_cond_frames,
                    num_end_cond=num_end_cond_frames,
                    transition_frames=normalize_transition_frames,
                    clump_values=True
                )
                print(f">>> Applied normalization with {normalize_transition_frames} transition frames at each end")

    # Save latents if requested
    if save_latents:
        latent_checkpoint = {
            "latents": latent_visual.cpu(),
            "shape": (bs, total_frames, height, width, dim),
            "mode": "v2v_join",
            "vae_scaling_factor": vae.config.scaling_factor,
            "latents_dtype": str(latent_visual.dtype),
        }
        torch.save(latent_checkpoint, save_latents)
        print(f">>> Latents saved to {save_latents}", flush=True)

    # Offload DiT before VAE decode
    if hasattr(dit, 'offload_all_blocks'):
        dit.offload_all_blocks()

    if offload or force_offload:
        dit = dit.to('cpu', non_blocking=True)
    torch.cuda.empty_cache()

    log_vram_usage("AFTER DiT OFFLOAD, BEFORE VAE DECODE (V2V JOIN)", dit=dit, vae=vae, text_embedder=None)

    if offload or force_offload:
        vae = vae.to(vae_device, non_blocking=True)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            images = latent_visual.reshape(
                bs,
                -1,
                latent_visual.shape[-3],
                latent_visual.shape[-2],
                latent_visual.shape[-1],
            )
            images = images.to(device=vae_device)
            images = (images / vae.config.scaling_factor).permute(0, 4, 1, 2, 3)
            images = vae.decode(images).sample
            images = ((images.clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8)

    log_vram_usage("AFTER VAE DECODE, BEFORE OFFLOAD (V2V JOIN)", dit=dit, vae=vae, text_embedder=None)

    if offload or force_offload:
        vae = vae.to('cpu', non_blocking=True)
    torch.cuda.empty_cache()

    return images


def generate_sample_v2v(
    shape,
    caption,
    dit,
    vae,
    conf,
    text_embedder,
    cond_latents,
    num_steps=50,
    guidance_weight=5.0,
    scheduler_scale=1,
    negative_caption="",
    clip_prompt=None,
    seed=6554,
    device="cuda",
    vae_device="cuda",
    progress=True,
    offload=False,
    force_offload=False,
    previewer=None,
    preview_interval=None,
    preview_suffix=None,
    stop_check=None,
    checkpoint_path=None,
    save_latents=None,
    use_apg=False,
    apg_momentum=0.9,
    apg_norm_threshold=55.0,
    apg_parallel_scale=0.3,
):
    """
    Generate video continuation from conditioning latents using KV cache.

    Args:
        shape: Output shape (bs, duration, height, width, dim) - duration is for NEW frames only
        caption: Text prompt
        dit: DiT model
        vae: VAE model
        conf: Configuration
        text_embedder: Text embedder
        cond_latents: Conditioning latents [num_cond_frames, H, W, C]
        ...

    Returns:
        Generated video tensor
    """
    text_embedder.embedder.mode = "i2v"

    bs, duration, height, width, dim = shape
    num_cond_frames = cond_latents.shape[0]

    if duration == 1:
        type_of_content = "image"
    else:
        type_of_content = "video"

    with torch.no_grad():
        clip_texts = [clip_prompt] if clip_prompt else None
        bs_text_embed, text_cu_seqlens, attention_mask = text_embedder.encode(
            [caption], type_of_content=type_of_content, clip_texts=clip_texts
        )
        bs_null_text_embed, null_text_cu_seqlens, null_attention_mask = text_embedder.encode(
            [negative_caption], type_of_content=type_of_content
        )

    # Clean up text embedder
    if offload or force_offload:
        text_embedder = text_embedder.to('cpu')
    del text_embedder.embedder.model
    del text_embedder.clip_embedder.model
    del text_embedder
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    for key in bs_text_embed:
        bs_text_embed[key] = bs_text_embed[key].to(device=device)
        bs_null_text_embed[key] = bs_null_text_embed[key].to(device=device)
    text_cu_seqlens = text_cu_seqlens.to(device=device)[-1].item()
    null_text_cu_seqlens = null_text_cu_seqlens.to(device=device)[-1].item()
    attention_mask = attention_mask.to(device=device)
    null_attention_mask = null_attention_mask.to(device=device)

    # Visual RoPE positions for FULL sequence (cond + new frames)
    total_frames = num_cond_frames + duration
    visual_rope_pos = [
        torch.arange(total_frames),  # [0, 1, 2, 3, 4, 5, ..., 34] for full sequence
        torch.arange(height // conf.model.dit_params.patch_size[1]),
        torch.arange(width // conf.model.dit_params.patch_size[2]),
    ]
    text_rope_pos = torch.arange(text_cu_seqlens)
    null_text_rope_pos = torch.arange(null_text_cu_seqlens)

    log_vram_usage("BEFORE DiT INFERENCE (V2V)", dit=dit, vae=vae, text_embedder=None)

    if offload or force_offload:
        dit.to(device, non_blocking=True)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # Generate video with visual conditioning (like I2V but with multiple frames)
            print(f">>> Generating video continuation with {num_cond_frames} conditioning frames...", flush=True)
            result = generate_v2v(
                dit,
                device,
                (bs * total_frames, height, width, dim),  # Full sequence shape
                num_steps,
                bs_text_embed,
                bs_null_text_embed,
                visual_rope_pos,
                text_rope_pos,
                null_text_rope_pos,
                guidance_weight,
                scheduler_scale,
                cond_latents.to(device=device, dtype=torch.bfloat16),
                conf,
                seed=seed,
                progress=progress,
                attention_mask=attention_mask,
                null_attention_mask=null_attention_mask,
                previewer=previewer,
                preview_interval=preview_interval,
                preview_suffix=preview_suffix,
                stop_check=stop_check,
                use_apg=use_apg,
                apg_momentum=apg_momentum,
                apg_norm_threshold=apg_norm_threshold,
                apg_parallel_scale=apg_parallel_scale,
            )

    # Handle early stop results
    if isinstance(result, dict) and result.get("action"):
        action = result["action"]
        latent_visual = result["latents"]
        step = result["step"]
        total_steps = result["total_steps"]
        original_noise = result.get("original_noise")
        timesteps = result.get("timesteps")

        if action == "save" and checkpoint_path:
            checkpoint = {
                "latents": latent_visual.cpu(),
                "step": step,
                "total_steps": total_steps,
                "seed": seed,
                "text_embeds": {k: v.cpu() for k, v in bs_text_embed.items()},
                "null_text_embeds": {k: v.cpu() for k, v in bs_null_text_embed.items()},
                "visual_rope_pos": [v.cpu() for v in visual_rope_pos],
                "text_rope_pos": text_rope_pos.cpu() if hasattr(text_rope_pos, 'cpu') else text_rope_pos,
                "null_text_rope_pos": null_text_rope_pos.cpu() if hasattr(null_text_rope_pos, 'cpu') else null_text_rope_pos,
                "cond_latents": cond_latents.cpu(),
                "shape": shape,
                "guidance_weight": guidance_weight,
                "scheduler_scale": scheduler_scale,
                "mode": "v2v",
            }
            torch.save(checkpoint, checkpoint_path)
            print(f">>> Checkpoint saved to {checkpoint_path} at step {step}/{total_steps}", flush=True)
            return None

        print(f">>> Decoding video from step {step}/{total_steps}", flush=True)

        if original_noise is not None and timesteps is not None:
            noise_remaining = timesteps[step]
            latent_visual = latent_visual - (original_noise * noise_remaining)
            print(f">>> Subtracted {noise_remaining.item():.4f} of original noise", flush=True)
    else:
        latent_visual = result

    # Note: No normalization needed - conditioning frames are preserved during denoising
    # The visual conditioning mechanism (setting img[:num_cond] = cond_latents)
    # ensures clean transition from conditioning to generated frames

    # Save latents if requested
    if save_latents:
        latent_checkpoint = {
            "latents": latent_visual.cpu(),
            "shape": (bs, num_cond_frames + duration, height, width, dim),
            "mode": "v2v",
            "vae_scaling_factor": vae.config.scaling_factor,
            "latents_dtype": str(latent_visual.dtype),
        }
        torch.save(latent_checkpoint, save_latents)
        print(f">>> Latents saved to {save_latents}", flush=True)

    # Offload DiT before VAE decode
    if hasattr(dit, 'offload_all_blocks'):
        dit.offload_all_blocks()

    if offload or force_offload:
        dit = dit.to('cpu', non_blocking=True)
    torch.cuda.empty_cache()

    log_vram_usage("AFTER DiT OFFLOAD, BEFORE VAE DECODE (V2V)", dit=dit, vae=vae, text_embedder=None)

    if offload or force_offload:
        vae = vae.to(vae_device, non_blocking=True)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            images = latent_visual.reshape(
                bs,
                -1,
                latent_visual.shape[-3],
                latent_visual.shape[-2],
                latent_visual.shape[-1],
            )
            images = images.to(device=vae_device)
            images = (images / vae.config.scaling_factor).permute(0, 4, 1, 2, 3)
            images = vae.decode(images).sample
            images = ((images.clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8)

    log_vram_usage("AFTER VAE DECODE, BEFORE OFFLOAD (V2V)", dit=dit, vae=vae, text_embedder=None)

    if offload or force_offload:
        vae = vae.to('cpu', non_blocking=True)
    torch.cuda.empty_cache()

    return images


def generate_sample_i2v(
    shape,
    caption,
    dit,
    vae,
    conf,
    text_embedder,
    images,
    num_steps=50,
    guidance_weight=5.0,
    scheduler_scale=1,
    negative_caption="",
    clip_prompt=None,
    seed=6554,
    device="cuda",
    vae_device="cuda",
    progress=True,
    offload=False,
    force_offload=False,
    previewer=None,
    preview_interval=None,
    preview_suffix=None,
    stop_check=None,
    checkpoint_path=None,
    save_latents=None,
):
    text_embedder.embedder.mode = "i2v"

    bs, duration, height, width, dim = shape
    if duration == 1:
        type_of_content = "image"
    else:
        type_of_content = "video"

    with torch.no_grad():
        # Pass clip_texts if a separate clip_prompt is provided
        clip_texts = [clip_prompt] if clip_prompt else None
        bs_text_embed, text_cu_seqlens, attention_mask = text_embedder.encode(
            [caption], type_of_content=type_of_content, clip_texts=clip_texts
        )
        bs_null_text_embed, null_text_cu_seqlens, null_attention_mask = text_embedder.encode(
            [negative_caption], type_of_content=type_of_content
        )

    # Clean up text embedder after encoding to free VRAM and RAM
    # Text embedder is no longer needed after this point
    if offload or force_offload:
        text_embedder = text_embedder.to('cpu')
    # Delete text embedder components to free memory
    del text_embedder.embedder.model
    del text_embedder.clip_embedder.model
    del text_embedder
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    for key in bs_text_embed:
        bs_text_embed[key] = bs_text_embed[key].to(device=device)
        bs_null_text_embed[key] = bs_null_text_embed[key].to(device=device)
    text_cu_seqlens = text_cu_seqlens.to(device=device)[-1].item()
    null_text_cu_seqlens = null_text_cu_seqlens.to(device=device)[-1].item()
    attention_mask = attention_mask.to(device=device)
    null_attention_mask = null_attention_mask.to(device=device)

    visual_rope_pos = [
        torch.arange(duration),
        torch.arange(shape[-3] // conf.model.dit_params.patch_size[1]),
        torch.arange(shape[-2] // conf.model.dit_params.patch_size[2]),
    ]
    text_rope_pos = torch.arange(text_cu_seqlens)
    null_text_rope_pos = torch.arange(null_text_cu_seqlens)

    # Log VRAM before DiT inference
    log_vram_usage("BEFORE DiT INFERENCE (I2V)", dit=dit, vae=vae, text_embedder=None)

    if offload or force_offload:
        dit.to(device, non_blocking=True)

    # Store first_frames for checkpoint
    first_frames = images

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            result = generate(
                dit,
                device,
                (bs * duration, height, width, dim),
                num_steps,
                bs_text_embed,
                bs_null_text_embed,
                visual_rope_pos,
                text_rope_pos,
                null_text_rope_pos,
                guidance_weight,
                scheduler_scale,
                images,
                conf,
                seed=seed,
                progress=progress,
                attention_mask=attention_mask,
                null_attention_mask=null_attention_mask,
                previewer=previewer,
                preview_interval=preview_interval,
                preview_suffix=preview_suffix,
                stop_check=stop_check,
            )

    # Handle early stop results
    if isinstance(result, dict) and result.get("action"):
        action = result["action"]
        latent_visual = result["latents"]
        step = result["step"]
        total_steps = result["total_steps"]
        original_noise = result.get("original_noise")
        timesteps = result.get("timesteps")

        if action == "save" and checkpoint_path:
            # Save checkpoint for later resumption
            checkpoint = {
                "latents": latent_visual.cpu(),
                "step": step,
                "total_steps": total_steps,
                "seed": seed,
                "text_embeds": {k: v.cpu() for k, v in bs_text_embed.items()},
                "null_text_embeds": {k: v.cpu() for k, v in bs_null_text_embed.items()},
                "visual_rope_pos": [v.cpu() for v in visual_rope_pos],
                "text_rope_pos": text_rope_pos.cpu() if hasattr(text_rope_pos, 'cpu') else text_rope_pos,
                "null_text_rope_pos": null_text_rope_pos.cpu() if hasattr(null_text_rope_pos, 'cpu') else null_text_rope_pos,
                "first_frames": first_frames.cpu() if first_frames is not None else None,
                "shape": shape,
                "guidance_weight": guidance_weight,
                "scheduler_scale": scheduler_scale,
                "mode": "i2v",
            }
            torch.save(checkpoint, checkpoint_path)
            print(f">>> Checkpoint saved to {checkpoint_path} at step {step}/{total_steps}", flush=True)
            return None  # Signal that we saved instead of decoding

        # For "decode" action, subtract remaining noise before decoding
        print(f">>> Decoding video from step {step}/{total_steps}", flush=True)

        # Subtract remaining noise (like preview does) to get clean latents
        if original_noise is not None and timesteps is not None:
            # timesteps[step] gives the noise level at the current step
            # We need to subtract this portion of the original noise
            noise_remaining = timesteps[step]
            latent_visual = latent_visual - (original_noise * noise_remaining)
            print(f">>> Subtracted {noise_remaining.item():.4f} of original noise", flush=True)
    else:
        latent_visual = result

    # Apply first frame normalization for i2v
    # Normalize generated frames (1-4) to match the input image (frame 0)
    # This ensures smooth transition from input image to generated content
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if first_frames is not None:
                first_frames = first_frames.to(device=latent_visual.device, dtype=latent_visual.dtype)
                latent_visual[:1] = first_frames
                # Normalize frames 1-4 to match frame 0 (the input image) for smooth transition
                latent_visual = normalize_generated_frames_to_conditioning(
                    latent_visual, num_cond_frames=1, transition_frames=4, clump_values=True
                )

    # Save latents before VAE decoding if requested
    if save_latents:
        latent_checkpoint = {
            "latents": latent_visual.cpu(),
            "shape": shape,
            "mode": "i2v",
            "vae_scaling_factor": vae.config.scaling_factor,
            "latents_dtype": str(latent_visual.dtype),
        }
        torch.save(latent_checkpoint, save_latents)
        print(f">>> Latents saved to {save_latents}", flush=True)

    # Offload DiT before VAE decode to free up VRAM
    # For block swapping, explicitly offload all blocks first
    if hasattr(dit, 'offload_all_blocks'):
        dit.offload_all_blocks()

    if offload or force_offload:
        dit = dit.to('cpu', non_blocking=True)
    torch.cuda.empty_cache()

    # Log VRAM after DiT offload, before VAE decode
    log_vram_usage("AFTER DiT OFFLOAD, BEFORE VAE DECODE (I2V)", dit=dit, vae=vae, text_embedder=None)

    if offload or force_offload:
        vae = vae.to(vae_device, non_blocking=True)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            images = latent_visual.reshape(
                bs,
                -1,
                latent_visual.shape[-3],
                latent_visual.shape[-2],
                latent_visual.shape[-1],
            )
            images = images.to(device=vae_device)

            # DEBUG: Check latent values before VAE decode
            print(f"\n{'='*80}", flush=True)
            print(f"DEBUG: Latents before VAE decode", flush=True)
            print(f"Shape: {images.shape}", flush=True)
            print(f"Dtype: {images.dtype}", flush=True)
            print(f"Device: {images.device}", flush=True)
            print(f"Min: {images.min().item():.6f}, Max: {images.max().item():.6f}", flush=True)
            print(f"Mean: {images.mean().item():.6f}, Std: {images.std().item():.6f}", flush=True)
            print(f"Has NaN: {torch.isnan(images).any().item()}", flush=True)
            print(f"Has Inf: {torch.isinf(images).any().item()}", flush=True)
            print(f"VAE scaling_factor: {vae.config.scaling_factor}", flush=True)
            print(f"VAE dtype: {next(vae.parameters()).dtype}", flush=True)
            print(f"{'='*80}\n", flush=True)

            images = (images / vae.config.scaling_factor).permute(0, 4, 1, 2, 3)

            print(f"\n{'='*80}", flush=True)
            print(f"DEBUG: After permute, ready for VAE", flush=True)
            print(f"Shape: {images.shape}", flush=True)
            print(f"Dtype: {images.dtype}", flush=True)
            print(f"Min: {images.min().item():.6f}, Max: {images.max().item():.6f}", flush=True)
            print(f"{'='*80}\n", flush=True)

            try:
                images = vae.decode(images).sample
                images = ((images.clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8)
            except Exception as e:
                print(f"\n{'='*80}", flush=True)
                print(f"ERROR during VAE decode:", flush=True)
                print(f"Exception type: {type(e).__name__}", flush=True)
                print(f"Exception message: {str(e)}", flush=True)
                print(f"{'='*80}\n", flush=True)
                import traceback
                traceback.print_exc()
                raise

    # Log VRAM after VAE decode, before VAE offload
    log_vram_usage("AFTER VAE DECODE, BEFORE OFFLOAD (I2V)", dit=dit, vae=vae, text_embedder=None)

    # Offload VAE after decode to free VRAM
    if offload or force_offload:
        vae = vae.to('cpu', non_blocking=True)
    torch.cuda.empty_cache()

    return images


@torch.no_grad()
def generate_resume(
    model,
    device,
    img,
    start_step,
    num_steps,
    text_embeds,
    null_text_embeds,
    visual_rope_pos,
    text_rope_pos,
    null_text_rope_pos,
    guidance_weight,
    scheduler_scale,
    first_frames,
    conf,
    progress=False,
    attention_mask=None,
    null_attention_mask=None,
    previewer=None,
    preview_interval=None,
    preview_suffix=None,
    stop_check=None,
):
    """Resume generation from a given step with pre-computed latents."""
    sparse_params = get_sparse_params(conf, {"visual": img}, device)
    timesteps = torch.linspace(1, 0, num_steps + 1, device=device)
    timesteps = scheduler_scale * timesteps / (1 + (scheduler_scale - 1) * timesteps)

    # Start from the saved step
    remaining_timesteps = list(zip(timesteps[start_step:-1], torch.diff(timesteps)[start_step:]))

    print(f">>> Resuming generation from step {start_step}/{num_steps}", flush=True)

    for i, (timestep, timestep_diff) in enumerate(tqdm(remaining_timesteps)):
        actual_step = start_step + i
        time = timestep.unsqueeze(0)
        if model.visual_cond:
            visual_cond = torch.zeros_like(img)
            visual_cond_mask = torch.zeros(
                [*img.shape[:-1], 1], dtype=img.dtype, device=img.device
            )
            if first_frames is not None:
                first_frames = first_frames.to(device=visual_cond.device, dtype=visual_cond.dtype)
                img[:1] = first_frames
                visual_cond_mask[:1] = 1
            model_input = torch.cat([img, visual_cond, visual_cond_mask], dim=-1)
        else:
            model_input = img
        pred_velocity = get_velocity(
            model,
            model_input,
            time,
            text_embeds,
            null_text_embeds,
            visual_rope_pos,
            text_rope_pos,
            null_text_rope_pos,
            guidance_weight,
            conf,
            sparse_params=sparse_params,
            attention_mask=attention_mask,
            null_attention_mask=null_attention_mask,
        )
        img = img + timestep_diff * pred_velocity

        # Check for early stop request
        if stop_check is not None:
            action = stop_check()
            if action in ("decode", "save"):
                print(f"\n>>> Early stop requested at step {actual_step + 1}/{num_steps} - action: {action}", flush=True)
                # Note: original_noise not available for resumed generation
                # Noise subtraction won't work for early-stop decode from checkpoints
                return {
                    "action": action,
                    "latents": img,
                    "step": actual_step + 1,
                    "total_steps": num_steps,
                    "original_noise": None,
                    "timesteps": timesteps
                }

        if previewer is not None and preview_interval and (actual_step + 1) % preview_interval == 0 and (actual_step + 1) < num_steps:
            import sys
            print(f"\n>>> PREVIEW TRIGGER at step {actual_step + 1}/{num_steps} (interval={preview_interval})", flush=True)
            sys.stdout.flush()
            try:
                preview_latent = img.permute(3, 0, 1, 2).unsqueeze(0)
                previewer.preview(preview_latent.squeeze(0), actual_step, preview_suffix=preview_suffix)
            except Exception as e:
                print(f">>> ERROR during preview generation at step {actual_step + 1}: {e}", flush=True)

    return img


def generate_sample_from_checkpoint(
    checkpoint_path,
    dit,
    vae,
    conf,
    device="cuda",
    vae_device="cuda",
    progress=True,
    offload=False,
    force_offload=False,
    image_vae=False,
    previewer=None,
    preview_interval=None,
    preview_suffix=None,
    stop_check=None,
    new_checkpoint_path=None,
):
    """Resume T2V generation from a saved checkpoint."""
    print(f">>> Loading checkpoint from {checkpoint_path}", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract checkpoint data
    img = checkpoint["latents"].to(device)
    start_step = checkpoint["step"]
    num_steps = checkpoint["total_steps"]
    bs_text_embed = {k: v.to(device, dtype=torch.bfloat16) for k, v in checkpoint["text_embeds"].items()}
    bs_null_text_embed = {k: v.to(device, dtype=torch.bfloat16) for k, v in checkpoint["null_text_embeds"].items()}
    visual_rope_pos = [v.to(device) if hasattr(v, 'to') else v for v in checkpoint["visual_rope_pos"]]
    text_rope_pos = checkpoint["text_rope_pos"]
    null_text_rope_pos = checkpoint["null_text_rope_pos"]
    if hasattr(text_rope_pos, 'to'):
        text_rope_pos = text_rope_pos.to(device)
    if hasattr(null_text_rope_pos, 'to'):
        null_text_rope_pos = null_text_rope_pos.to(device)
    shape = checkpoint["shape"]
    guidance_weight = checkpoint["guidance_weight"]
    scheduler_scale = checkpoint["scheduler_scale"]

    bs = shape[0]

    # Log VRAM before DiT inference
    log_vram_usage("BEFORE DiT INFERENCE (RESUME T2V)", dit=dit, vae=vae, text_embedder=None)

    if offload or force_offload:
        dit.to(device, non_blocking=True)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            result = generate_resume(
                dit,
                device,
                img,
                start_step,
                num_steps,
                bs_text_embed,
                bs_null_text_embed,
                visual_rope_pos,
                text_rope_pos,
                null_text_rope_pos,
                guidance_weight,
                scheduler_scale,
                None,  # first_frames (T2V doesn't use this)
                conf,
                progress=progress,
                previewer=previewer,
                preview_interval=preview_interval,
                preview_suffix=preview_suffix,
                stop_check=stop_check,
            )

    # Handle early stop results
    if isinstance(result, dict) and result.get("action"):
        action = result["action"]
        latent_visual = result["latents"]
        step = result["step"]
        total_steps = result["total_steps"]

        if action == "save" and new_checkpoint_path:
            # Save new checkpoint
            new_checkpoint = {
                "latents": latent_visual.cpu(),
                "step": step,
                "total_steps": total_steps,
                "seed": checkpoint.get("seed", 0),
                "text_embeds": {k: v.cpu() for k, v in bs_text_embed.items()},
                "null_text_embeds": {k: v.cpu() for k, v in bs_null_text_embed.items()},
                "visual_rope_pos": [v.cpu() if hasattr(v, 'cpu') else v for v in visual_rope_pos],
                "text_rope_pos": text_rope_pos.cpu() if hasattr(text_rope_pos, 'cpu') else text_rope_pos,
                "null_text_rope_pos": null_text_rope_pos.cpu() if hasattr(null_text_rope_pos, 'cpu') else null_text_rope_pos,
                "shape": shape,
                "guidance_weight": guidance_weight,
                "scheduler_scale": scheduler_scale,
            }
            torch.save(new_checkpoint, new_checkpoint_path)
            print(f">>> Checkpoint saved to {new_checkpoint_path} at step {step}/{total_steps}", flush=True)
            return None

        print(f">>> Decoding video from step {step}/{total_steps}", flush=True)
    else:
        latent_visual = result

    # Offload DiT before VAE decode
    if hasattr(dit, 'offload_all_blocks'):
        dit.offload_all_blocks()

    if offload or force_offload:
        dit = dit.to('cpu', non_blocking=True)
    torch.cuda.empty_cache()

    if offload or force_offload:
        vae = vae.to(vae_device, non_blocking=True)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            images = latent_visual.reshape(
                bs,
                -1,
                latent_visual.shape[-3],
                latent_visual.shape[-2],
                latent_visual.shape[-1],
            )
            images = images.to(device=vae_device)
            images = (images / vae.config.scaling_factor).permute(0, 4, 1, 2, 3)
            if image_vae:
                images = images[:,:,0]
            images = vae.decode(images).sample
            images = ((images.clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8)

    if offload or force_offload:
        vae = vae.to('cpu', non_blocking=True)
    torch.cuda.empty_cache()

    return images


def generate_sample_i2v_from_checkpoint(
    checkpoint_path,
    dit,
    vae,
    conf,
    device="cuda",
    vae_device="cuda",
    progress=True,
    offload=False,
    force_offload=False,
    previewer=None,
    preview_interval=None,
    preview_suffix=None,
    stop_check=None,
    new_checkpoint_path=None,
):
    """Resume I2V generation from a saved checkpoint."""
    print(f">>> Loading checkpoint from {checkpoint_path}", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract checkpoint data
    img = checkpoint["latents"].to(device)
    start_step = checkpoint["step"]
    num_steps = checkpoint["total_steps"]
    bs_text_embed = {k: v.to(device, dtype=torch.bfloat16) for k, v in checkpoint["text_embeds"].items()}
    bs_null_text_embed = {k: v.to(device, dtype=torch.bfloat16) for k, v in checkpoint["null_text_embeds"].items()}
    visual_rope_pos = [v.to(device) if hasattr(v, 'to') else v for v in checkpoint["visual_rope_pos"]]
    text_rope_pos = checkpoint["text_rope_pos"]
    null_text_rope_pos = checkpoint["null_text_rope_pos"]
    if hasattr(text_rope_pos, 'to'):
        text_rope_pos = text_rope_pos.to(device)
    if hasattr(null_text_rope_pos, 'to'):
        null_text_rope_pos = null_text_rope_pos.to(device)
    first_frames = checkpoint.get("first_frames")
    if first_frames is not None:
        first_frames = first_frames.to(device)
    shape = checkpoint["shape"]
    guidance_weight = checkpoint["guidance_weight"]
    scheduler_scale = checkpoint["scheduler_scale"]

    bs = shape[0]

    # Log VRAM before DiT inference
    log_vram_usage("BEFORE DiT INFERENCE (RESUME I2V)", dit=dit, vae=vae, text_embedder=None)

    if offload or force_offload:
        dit.to(device, non_blocking=True)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            result = generate_resume(
                dit,
                device,
                img,
                start_step,
                num_steps,
                bs_text_embed,
                bs_null_text_embed,
                visual_rope_pos,
                text_rope_pos,
                null_text_rope_pos,
                guidance_weight,
                scheduler_scale,
                first_frames,
                conf,
                progress=progress,
                previewer=previewer,
                preview_interval=preview_interval,
                preview_suffix=preview_suffix,
                stop_check=stop_check,
            )

    # Handle early stop results
    if isinstance(result, dict) and result.get("action"):
        action = result["action"]
        latent_visual = result["latents"]
        step = result["step"]
        total_steps = result["total_steps"]

        if action == "save" and new_checkpoint_path:
            # Save new checkpoint
            new_checkpoint = {
                "latents": latent_visual.cpu(),
                "step": step,
                "total_steps": total_steps,
                "seed": checkpoint.get("seed", 0),
                "text_embeds": {k: v.cpu() for k, v in bs_text_embed.items()},
                "null_text_embeds": {k: v.cpu() for k, v in bs_null_text_embed.items()},
                "visual_rope_pos": [v.cpu() if hasattr(v, 'cpu') else v for v in visual_rope_pos],
                "text_rope_pos": text_rope_pos.cpu() if hasattr(text_rope_pos, 'cpu') else text_rope_pos,
                "null_text_rope_pos": null_text_rope_pos.cpu() if hasattr(null_text_rope_pos, 'cpu') else null_text_rope_pos,
                "first_frames": first_frames.cpu() if first_frames is not None else None,
                "shape": shape,
                "guidance_weight": guidance_weight,
                "scheduler_scale": scheduler_scale,
                "mode": "i2v",
            }
            torch.save(new_checkpoint, new_checkpoint_path)
            print(f">>> Checkpoint saved to {new_checkpoint_path} at step {step}/{total_steps}", flush=True)
            return None

        print(f">>> Decoding video from step {step}/{total_steps}", flush=True)
    else:
        latent_visual = result

    # Apply first frame normalization for i2v
    # Normalize generated frames (1-4) to match the input image (frame 0)
    # This ensures smooth transition from input image to generated content
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if first_frames is not None:
                first_frames = first_frames.to(device=latent_visual.device, dtype=latent_visual.dtype)
                latent_visual[:1] = first_frames
                # Normalize frames 1-4 to match frame 0 (the input image) for smooth transition
                latent_visual = normalize_generated_frames_to_conditioning(
                    latent_visual, num_cond_frames=1, transition_frames=4, clump_values=True
                )

    # Offload DiT before VAE decode
    if hasattr(dit, 'offload_all_blocks'):
        dit.offload_all_blocks()

    if offload or force_offload:
        dit = dit.to('cpu', non_blocking=True)
    torch.cuda.empty_cache()

    if offload or force_offload:
        vae = vae.to(vae_device, non_blocking=True)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            images = latent_visual.reshape(
                bs,
                -1,
                latent_visual.shape[-3],
                latent_visual.shape[-2],
                latent_visual.shape[-1],
            )
            images = images.to(device=vae_device)
            images = (images / vae.config.scaling_factor).permute(0, 4, 1, 2, 3)
            images = vae.decode(images).sample
            images = ((images.clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8)

    if offload or force_offload:
        vae = vae.to('cpu', non_blocking=True)
    torch.cuda.empty_cache()

    return images
