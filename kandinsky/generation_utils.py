import os
os.environ["TOKENIZERS_PARALLELISM"] = "False"

import time
import torch
from tqdm import tqdm

from .models.utils import fast_sta_nabla

# APG (Adaptive Projected Guidance) helpers for reducing color drift in long video generation
# Reference: https://arxiv.org/abs/2410.02416

class MomentumBuffer:
    """Momentum buffer for APG to smooth guidance updates."""
    def __init__(self, momentum: float):
        self.momentum = momentum
        self.running_average = 0

    def update(self, update_value: torch.Tensor):
        new_average = self.momentum * self.running_average
        self.running_average = update_value + new_average


def project(v0: torch.Tensor, v1: torch.Tensor):
    """Project v0 onto v1 and get orthogonal component."""
    dtype = v0.dtype
    v0, v1 = v0.double(), v1.double()
    v1 = torch.nn.functional.normalize(v1, dim=[-1, -2, -3, -4])
    v0_parallel = (v0 * v1).sum(dim=[-1, -2, -3, -4], keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    return v0_parallel.to(dtype), v0_orthogonal.to(dtype)


def adaptive_projected_guidance(
    diff: torch.Tensor,
    pred_cond: torch.Tensor,
    momentum_buffer: MomentumBuffer = None,
    norm_threshold: float = 55,
):
    """
    Apply Adaptive Projected Guidance to reduce color drift.

    Args:
        diff: Difference between conditional and unconditional predictions
        pred_cond: Conditional prediction
        momentum_buffer: Optional momentum buffer for smoothing
        norm_threshold: Threshold for norm clipping (0 to disable)

    Returns:
        Adjusted guidance direction (orthogonal component)
    """
    if momentum_buffer is not None:
        momentum_buffer.update(diff)
        diff = momentum_buffer.running_average
    if norm_threshold > 0:
        ones = torch.ones_like(diff)
        diff_norm = diff.norm(p=2, dim=[-1, -2, -3, -4], keepdim=True)
        scale_factor = torch.minimum(ones, norm_threshold / diff_norm)
        diff = diff * scale_factor
    diff_parallel, diff_orthogonal = project(diff, pred_cond)
    return diff_orthogonal


def adaptive_mean_std_normalization(source, reference, clump_mean_low=0.3, clump_mean_high=0.35, clump_std_low=0.35, clump_std_high=0.5):
    # source shape is [frames, H, W, C] - 4D tensor
    source_mean = source.mean(dim=(1, 2, 3), keepdim=True)  # mean over H, W, C
    source_std = source.std(dim=(1, 2, 3), keepdim=True)    # std over H, W, C

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
    # Pipeline parallelism parameters:
    start_step=0,           # Start from this step (0-indexed)
    end_step=None,          # End at this step (None = num_steps)
    initial_latent=None,    # Continue from this latent instead of random noise
):
    # Handle pipeline parallelism - use provided latent or generate new noise
    if initial_latent is not None:
        img = initial_latent.to(device)
    else:
        g = torch.Generator(device="cuda")
        g.manual_seed(seed)
        img = torch.randn(*shape, device=device, generator=g)

    # Store original noise for early-stop decode
    original_noise = img.clone()

    sparse_params = get_sparse_params(conf, {"visual": img}, device)
    timesteps = torch.linspace(1, 0, num_steps + 1, device=device)
    timesteps = scheduler_scale * timesteps / (1 + (scheduler_scale - 1) * timesteps)

    # Handle step range for pipeline parallelism
    if end_step is None:
        end_step = num_steps

    # Create step pairs for the specified range
    step_pairs = list(zip(timesteps[:-1], torch.diff(timesteps)))[start_step:end_step]

    for i, (timestep, timestep_diff) in enumerate(tqdm(step_pairs, initial=start_step, total=end_step)):
        actual_step = start_step + i  # Track actual step number for logging
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
                return {
                    "action": action,
                    "latents": img,
                    "step": actual_step + 1,
                    "total_steps": num_steps,
                    "original_noise": original_noise,
                    "timesteps": timesteps
                }

        if previewer is not None and preview_interval and (actual_step + 1) % preview_interval == 0 and (actual_step + 1) < num_steps:
            import sys
            print(f"\n>>> PREVIEW TRIGGER at step {actual_step + 1}/{num_steps} (interval={preview_interval})", flush=True)
            sys.stdout.flush()
            print(f">>> img shape before permute: {img.shape}", flush=True)
            try:
                preview_latent = img.permute(3, 0, 1, 2).unsqueeze(0)
                print(f">>> preview_latent shape after permute+unsqueeze: {preview_latent.shape}", flush=True)
                previewer.preview(preview_latent.squeeze(0), actual_step, preview_suffix=preview_suffix)
                print(f">>> Preview completed successfully", flush=True)
                sys.stdout.flush()
            except Exception as e:
                print(f">>> ERROR during preview generation at step {actual_step + 1}: {e}", flush=True)
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
        print("[TIMING] Moving text encoder to GPU...", flush=True)
        t0 = time.perf_counter()
        text_embedder = text_embedder.to(device)
        print(f"[TIMING] Text encoder on GPU in {time.perf_counter() - t0:.1f}s", flush=True)

        t0 = time.perf_counter()
        clip_texts = [clip_prompt] if clip_prompt else None
        bs_text_embed, text_cu_seqlens, attention_mask = text_embedder.encode(
            [caption], type_of_content=type_of_content, clip_texts=clip_texts
        )
        bs_null_text_embed, null_text_cu_seqlens, null_attention_mask = text_embedder.encode(
            [negative_caption], type_of_content=type_of_content
        )
        print(f"[TIMING] Text encoding done in {time.perf_counter() - t0:.1f}s", flush=True)

    # Offload text encoder to CPU before DiT inference to free VRAM
    print("[TIMING] Offloading text encoder to CPU...", flush=True)
    t0 = time.perf_counter()
    text_embedder = text_embedder.to('cpu')
    torch.cuda.empty_cache()
    print(f"[TIMING] Text encoder offloaded in {time.perf_counter() - t0:.1f}s", flush=True)

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
    apg_momentum=-0.75,
    apg_norm_threshold=55.0,
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
                        diff_reshaped, pred_reshaped, momentum_buffer, apg_norm_threshold
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
                preview_latent = img.permute(3, 0, 1, 2).unsqueeze(0)
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
    apg_momentum=-0.75,
    apg_norm_threshold=55.0,
    # Noise scheduling parameters for end frames
    end_noise_schedule="progressive",  # "progressive", "fixed", "symmetric"
    end_noise_start=1.0,  # Initial noise level for end frames (1.0 = full noise, 0.0 = clean)
    end_noise_end=0.0,    # Final noise level for end frames
    start_noise_schedule="fixed",  # Usually keep start frames clean
    start_noise_level=0.0,  # Noise level for start frames (0.0 = clean)
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
        end_noise_schedule: Noise schedule for end frames:
            - "progressive": Noise decreases from end_noise_start to end_noise_end over steps
            - "fixed": Use constant noise level (end_noise_start)
            - "symmetric": Match the timestep (same as middle region would have)
        end_noise_start: Initial noise level for end frames (1.0 = full noise, 0.0 = clean)
        end_noise_end: Final noise level for end frames (only for "progressive")
        start_noise_schedule: Noise schedule for start frames (usually "fixed")
        start_noise_level: Noise level for start frames (0.0 = clean anchoring)
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

    # Store original noise for early-stop decode
    original_noise = img.clone()

    # Store noise for conditioning frames to enable noise injection
    start_noise = img[:num_start_cond_frames].clone()
    end_noise = img[-num_end_cond_frames:].clone()

    sparse_params = get_sparse_params(conf, {"visual": img}, device)
    timesteps = torch.linspace(1, 0, num_steps + 1, device=device)
    timesteps = scheduler_scale * timesteps / (1 + (scheduler_scale - 1) * timesteps)

    # Pre-compute noise schedule for end frames
    if end_noise_schedule == "progressive":
        # Linear interpolation from end_noise_start to end_noise_end
        end_noise_levels = torch.linspace(end_noise_start, end_noise_end, num_steps, device=device)
    elif end_noise_schedule == "symmetric":
        # Use the same timestep as middle region
        end_noise_levels = timesteps[:-1].clone()
    else:  # "fixed"
        end_noise_levels = torch.full((num_steps,), end_noise_start, device=device)

    # Log noise schedule info
    print(f">>> End noise schedule: {end_noise_schedule}")
    print(f">>> End noise levels: start={end_noise_levels[0].item():.4f}, mid={end_noise_levels[num_steps//2].item():.4f}, end={end_noise_levels[-1].item():.4f}")

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

            # Apply noise scheduling to start frames
            if start_noise_schedule == "fixed" and start_noise_level > 0:
                noisy_start = start_cond_device + start_noise_level * start_noise
                img[:num_start_cond_frames] = noisy_start
            else:
                img[:num_start_cond_frames] = start_cond_device

            # Apply noise scheduling to end frames
            end_noise_level = end_noise_levels[i]
            if end_noise_level > 0:
                # Add noise proportional to the schedule: x_noisy = x_clean + sigma * noise
                noisy_end = end_cond_device + end_noise_level * end_noise
                img[-num_end_cond_frames:] = noisy_end
            else:
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
                        diff_reshaped, pred_reshaped, momentum_buffer, apg_norm_threshold
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
                # Preview all frames being processed (conditioning + generated)
                preview_latent = img.permute(3, 0, 1, 2).unsqueeze(0)
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
    apg_momentum=-0.75,
    apg_norm_threshold=55.0,
    # Noise scheduling parameters
    end_noise_schedule="progressive",
    end_noise_start=1.0,
    end_noise_end=0.0,
    start_noise_schedule="fixed",
    start_noise_level=0.0,
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
        print("[TIMING] Moving text encoder to GPU...", flush=True)
        t0 = time.perf_counter()
        text_embedder = text_embedder.to(device)
        print(f"[TIMING] Text encoder on GPU in {time.perf_counter() - t0:.1f}s", flush=True)

        t0 = time.perf_counter()
        clip_texts = [clip_prompt] if clip_prompt else None
        bs_text_embed, text_cu_seqlens, attention_mask = text_embedder.encode(
            [caption], type_of_content=type_of_content, clip_texts=clip_texts
        )
        bs_null_text_embed, null_text_cu_seqlens, null_attention_mask = text_embedder.encode(
            [negative_caption], type_of_content=type_of_content
        )
        print(f"[TIMING] Text encoding done in {time.perf_counter() - t0:.1f}s", flush=True)

    # Delete text encoder to free VRAM
    # gc.collect() must run BEFORE empty_cache() to release Python references first
    print("[TIMING] Deleting text encoder from VRAM...", flush=True)
    t0 = time.perf_counter()
    del text_embedder.embedder.model
    del text_embedder.clip_embedder.model
    del text_embedder
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[TIMING] Text encoder deleted in {time.perf_counter() - t0:.1f}s", flush=True)

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
                # Pass noise scheduling parameters
                end_noise_schedule=end_noise_schedule,
                end_noise_start=end_noise_start,
                end_noise_end=end_noise_end,
                start_noise_schedule=start_noise_schedule,
                start_noise_level=start_noise_level,
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
    apg_momentum=-0.75,
    apg_norm_threshold=55.0,
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
        print("[TIMING] Moving text encoder to GPU...", flush=True)
        t0 = time.perf_counter()
        text_embedder = text_embedder.to(device)
        print(f"[TIMING] Text encoder on GPU in {time.perf_counter() - t0:.1f}s", flush=True)

        t0 = time.perf_counter()
        clip_texts = [clip_prompt] if clip_prompt else None
        bs_text_embed, text_cu_seqlens, attention_mask = text_embedder.encode(
            [caption], type_of_content=type_of_content, clip_texts=clip_texts
        )
        bs_null_text_embed, null_text_cu_seqlens, null_attention_mask = text_embedder.encode(
            [negative_caption], type_of_content=type_of_content
        )
        print(f"[TIMING] Text encoding done in {time.perf_counter() - t0:.1f}s", flush=True)

    # Delete text encoder to free VRAM
    # gc.collect() must run BEFORE empty_cache() to release Python references first
    print("[TIMING] Deleting text encoder from VRAM...", flush=True)
    t0 = time.perf_counter()
    del text_embedder.embedder.model
    del text_embedder.clip_embedder.model
    del text_embedder
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[TIMING] Text encoder deleted in {time.perf_counter() - t0:.1f}s", flush=True)

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
        print("[TIMING] Moving text encoder to GPU...", flush=True)
        t0 = time.perf_counter()
        text_embedder = text_embedder.to(device)
        print(f"[TIMING] Text encoder on GPU in {time.perf_counter() - t0:.1f}s", flush=True)

        t0 = time.perf_counter()
        clip_texts = [clip_prompt] if clip_prompt else None
        bs_text_embed, text_cu_seqlens, attention_mask = text_embedder.encode(
            [caption], type_of_content=type_of_content, clip_texts=clip_texts
        )
        bs_null_text_embed, null_text_cu_seqlens, null_attention_mask = text_embedder.encode(
            [negative_caption], type_of_content=type_of_content
        )
        print(f"[TIMING] Text encoding done in {time.perf_counter() - t0:.1f}s", flush=True)

    # Offload text encoder to CPU before DiT inference to free VRAM
    print("[TIMING] Offloading text encoder to CPU...", flush=True)
    t0 = time.perf_counter()
    text_embedder = text_embedder.to('cpu')
    torch.cuda.empty_cache()
    print(f"[TIMING] Text encoder offloaded in {time.perf_counter() - t0:.1f}s", flush=True)

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
    log_vram_usage("BEFORE DiT INFERENCE (I2V)", dit=dit, vae=vae, text_embedder=text_embedder)

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
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if first_frames is not None:
                first_frames = first_frames.to(device=latent_visual.device, dtype=latent_visual.dtype)
                latent_visual[:1] = first_frames
                latent_visual = normalize_first_frame(latent_visual)

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


@torch.no_grad()
def generate_denoise(
    model,
    device,
    latents,
    start_timestep,
    num_steps,
    text_embeds,
    null_text_embeds,
    visual_rope_pos,
    text_rope_pos,
    null_text_rope_pos,
    guidance_weight,
    scheduler_scale,
    conf,
    progress=False,
    attention_mask=None,
    null_attention_mask=None,
    preserve_first_frame=True,
    previewer=None,
    preview_interval=None,
    preview_suffix=None,
):
    """
    Denoise latents starting from a given timestep (for v2v img2img style processing).

    Args:
        model: DiT model
        device: Device to use
        latents: Clean latents to denoise [frames, H, W, C]
        start_timestep: Starting timestep (0.0-1.0), lower = less noise added
        num_steps: Number of denoising steps to perform
        text_embeds: Text embeddings
        null_text_embeds: Null text embeddings for CFG
        visual_rope_pos: Visual RoPE positions
        text_rope_pos: Text RoPE positions
        null_text_rope_pos: Null text RoPE positions
        guidance_weight: CFG weight
        scheduler_scale: Scheduler scale
        conf: Model configuration
        preserve_first_frame: If True, don't add noise to first frame (preserves video quality)

    Returns:
        Denoised latents
    """
    sparse_params = get_sparse_params(conf, {"visual": latents}, device)

    # Create a schedule from start_timestep to 0 with num_steps steps
    # This gives us full control over how many steps to use
    raw_timesteps = torch.linspace(start_timestep, 0, num_steps + 1, device=device)

    # Apply scheduler scaling to warp the timesteps
    # The formula: scaled_t = scheduler_scale * t / (1 + (scheduler_scale - 1) * t)
    timesteps = scheduler_scale * raw_timesteps / (1 + (scheduler_scale - 1) * raw_timesteps)

    # Save the clean first frame if we need to preserve it
    first_frame_clean = latents[0:1].clone() if preserve_first_frame else None

    # Generate noise and create noisy latents at start_timestep
    # Flow matching: x_t = (1-t)*x_0 + t*noise
    noise = torch.randn_like(latents)
    t = start_timestep
    img = (1 - t) * latents + t * noise

    # Restore the first frame to clean (no noise) if preserving
    if preserve_first_frame:
        img[0:1] = first_frame_clean
        print(f">>> Denoising from timestep {start_timestep:.3f} with {num_steps} steps (first frame preserved)", flush=True)
    else:
        print(f">>> Denoising from timestep {start_timestep:.3f} with {num_steps} steps", flush=True)

    for i, (timestep, timestep_diff) in enumerate(tqdm(list(zip(timesteps[:-1], torch.diff(timesteps))))):
        time = timestep.unsqueeze(0)

        if model.visual_cond:
            visual_cond = torch.zeros_like(img)
            visual_cond_mask = torch.zeros(
                [*img.shape[:-1], 1], dtype=img.dtype, device=img.device
            )
            # Use visual conditioning for first frame to preserve it during denoising
            if preserve_first_frame:
                img[0:1] = first_frame_clean.to(device=img.device, dtype=img.dtype)
                visual_cond_mask[0:1] = 1
            model_input = torch.cat([img, visual_cond, visual_cond_mask], dim=-1)
        else:
            model_input = img

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
                pred_velocity = uncond_pred_velocity + guidance_weight * (
                    pred_velocity - uncond_pred_velocity
                )

        img = img + timestep_diff * pred_velocity

        # Generate preview if enabled
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

    # Ensure first frame is exactly preserved in final output
    if preserve_first_frame:
        img[0:1] = first_frame_clean.to(device=img.device, dtype=img.dtype)

    return img


def generate_sample_denoise(
    video_latents,
    caption,
    dit,
    vae,
    conf,
    text_embedder,
    denoise_strength=0.2,
    num_steps=50,
    guidance_weight=5.0,
    scheduler_scale=10.0,
    negative_caption="",
    clip_prompt=None,
    seed=6554,
    device="cuda",
    vae_device="cuda",
    progress=True,
    offload=False,
    force_offload=False,
    chunk_frames=31,
    chunk_overlap=4,
    previewer=None,
    preview_interval=None,
    preview_suffix=None,
):
    """
    Apply light denoising to video latents (v2v img2img style).

    This takes already-encoded video latents, adds a small amount of noise,
    and denoises them. Useful for smoothing artifacts at video join points.
    Processes video in chunks to handle videos longer than 5 seconds.

    Args:
        video_latents: Pre-encoded video latents [frames, H, W, C]
        caption: Text prompt describing the video
        dit: DiT model
        vae: VAE model
        conf: Configuration
        text_embedder: Text embedder
        denoise_strength: How much to denoise (0.1-0.5 typical). Higher = more change.
        num_steps: Base number of denoising steps
        guidance_weight: CFG weight (lower like 2-3 for preservation)
        scheduler_scale: Scheduler scale
        chunk_frames: Max frames per chunk (default 31 = 5 seconds)
        chunk_overlap: Overlap frames between chunks for blending (default 4)
        ...

    Returns:
        Denoised video tensor [1, C, frames, H, W] as uint8
    """
    text_embedder.embedder.mode = "i2v"

    total_frames = video_latents.shape[0]
    height, width = video_latents.shape[1:3]
    dim = video_latents.shape[3]

    type_of_content = "video"

    with torch.no_grad():
        print("[TIMING] Moving text encoder to GPU...", flush=True)
        t0 = time.perf_counter()
        text_embedder = text_embedder.to(device)
        print(f"[TIMING] Text encoder on GPU in {time.perf_counter() - t0:.1f}s", flush=True)

        t0 = time.perf_counter()
        clip_texts = [clip_prompt] if clip_prompt else None
        bs_text_embed, text_cu_seqlens, attention_mask = text_embedder.encode(
            [caption], type_of_content=type_of_content, clip_texts=clip_texts
        )
        bs_null_text_embed, null_text_cu_seqlens, null_attention_mask = text_embedder.encode(
            [negative_caption], type_of_content=type_of_content
        )
        print(f"[TIMING] Text encoding done in {time.perf_counter() - t0:.1f}s", flush=True)

    # Delete text encoder to free VRAM
    # gc.collect() must run BEFORE empty_cache() to release Python references first
    print("[TIMING] Deleting text encoder from VRAM...", flush=True)
    t0 = time.perf_counter()
    del text_embedder.embedder.model
    del text_embedder.clip_embedder.model
    del text_embedder
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[TIMING] Text encoder deleted in {time.perf_counter() - t0:.1f}s", flush=True)

    for key in bs_text_embed:
        bs_text_embed[key] = bs_text_embed[key].to(device=device, dtype=torch.bfloat16)
        bs_null_text_embed[key] = bs_null_text_embed[key].to(device=device, dtype=torch.bfloat16)
    text_cu_seqlens = text_cu_seqlens.to(device=device)[-1].item()
    null_text_cu_seqlens = null_text_cu_seqlens.to(device=device)[-1].item()
    attention_mask = attention_mask.to(device=device)
    null_attention_mask = null_attention_mask.to(device=device)

    text_rope_pos = torch.arange(text_cu_seqlens)
    null_text_rope_pos = torch.arange(null_text_cu_seqlens)

    log_vram_usage("BEFORE DiT INFERENCE (DENOISE)", dit=dit, vae=vae, text_embedder=None)

    if offload or force_offload:
        dit.to(device, non_blocking=True)

    # Set random seed for reproducible noise
    torch.manual_seed(seed)

    # Process video in chunks
    # If video fits in one chunk, no need for multiple chunks
    if total_frames <= chunk_frames:
        num_chunks = 1
    else:
        num_chunks = (total_frames - chunk_overlap - 1) // (chunk_frames - chunk_overlap) + 1
        num_chunks = max(1, num_chunks)

    print(f">>> Denoising video with strength {denoise_strength}...", flush=True)
    print(f">>> Total frames: {total_frames}, processing in {num_chunks} chunk(s) of {chunk_frames} frames", flush=True)

    # Output buffer for denoised latents
    denoised_latents = torch.zeros_like(video_latents)
    blend_weights = torch.zeros(total_frames, 1, 1, 1, device=video_latents.device, dtype=video_latents.dtype)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            chunk_start = 0
            chunk_idx = 0

            while chunk_start < total_frames:
                chunk_end = min(chunk_start + chunk_frames, total_frames)
                actual_chunk_size = chunk_end - chunk_start

                print(f">>> Processing chunk {chunk_idx + 1}/{num_chunks}: frames {chunk_start}-{chunk_end} ({actual_chunk_size} frames)", flush=True)

                # Extract chunk
                chunk_latents = video_latents[chunk_start:chunk_end].to(device=device, dtype=torch.bfloat16)

                # Create visual RoPE positions for this chunk (always starts from 0)
                visual_rope_pos = [
                    torch.arange(actual_chunk_size),
                    torch.arange(height // conf.model.dit_params.patch_size[1]),
                    torch.arange(width // conf.model.dit_params.patch_size[2]),
                ]

                # Denoise this chunk
                # Only preserve the first frame when processing the first chunk (chunk_start == 0)
                # This ensures the original video's first frame is never noised
                chunk_result = generate_denoise(
                    dit,
                    device,
                    chunk_latents,
                    start_timestep=denoise_strength,
                    num_steps=num_steps,
                    text_embeds=bs_text_embed,
                    null_text_embeds=bs_null_text_embed,
                    visual_rope_pos=visual_rope_pos,
                    text_rope_pos=text_rope_pos,
                    null_text_rope_pos=null_text_rope_pos,
                    guidance_weight=guidance_weight,
                    scheduler_scale=scheduler_scale,
                    conf=conf,
                    progress=progress,
                    attention_mask=attention_mask,
                    null_attention_mask=null_attention_mask,
                    preserve_first_frame=(chunk_start == 0),
                    previewer=previewer,
                    preview_interval=preview_interval,
                    preview_suffix=preview_suffix,
                )

                # Move result back to CPU to save VRAM
                chunk_result = chunk_result.to(device=video_latents.device, dtype=video_latents.dtype)

                # Blend chunk into output with linear ramp for overlapping regions
                for i in range(actual_chunk_size):
                    frame_idx = chunk_start + i

                    # Calculate blend weight for this frame
                    # Ramp up at start of chunk (if not first chunk)
                    # Ramp down at end of chunk (if not last chunk)
                    weight = 1.0

                    if chunk_start > 0 and i < chunk_overlap:
                        # Ramp up: 0 to 1 over overlap region
                        weight = (i + 1) / (chunk_overlap + 1)
                    elif chunk_end < total_frames and i >= actual_chunk_size - chunk_overlap:
                        # Ramp down: 1 to 0 over overlap region
                        frames_from_end = actual_chunk_size - 1 - i
                        weight = (frames_from_end + 1) / (chunk_overlap + 1)

                    denoised_latents[frame_idx] += chunk_result[i] * weight
                    blend_weights[frame_idx] += weight

                # Clear GPU memory
                del chunk_latents, chunk_result
                torch.cuda.empty_cache()

                # Move to next chunk (with overlap)
                chunk_start = chunk_end - chunk_overlap
                if chunk_start >= total_frames - chunk_overlap:
                    break
                chunk_idx += 1

    # Normalize by blend weights
    blend_weights = blend_weights.clamp(min=1e-8)
    denoised_latents = denoised_latents / blend_weights

    latent_visual = denoised_latents

    # Offload DiT before VAE decode
    if hasattr(dit, 'offload_all_blocks'):
        dit.offload_all_blocks()

    if offload or force_offload:
        dit = dit.to('cpu', non_blocking=True)
    torch.cuda.empty_cache()

    log_vram_usage("AFTER DiT OFFLOAD, BEFORE VAE DECODE (DENOISE)", dit=dit, vae=vae, text_embedder=None)

    if offload or force_offload:
        vae = vae.to(vae_device, non_blocking=True)

    # Decode in chunks to avoid VAE OOM
    print(f">>> Decoding {total_frames} latent frames to video...", flush=True)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # VAE can handle more frames than DiT, but still chunk for safety
            vae_chunk_size = 31  # ~5 seconds of video at a time
            decoded_chunks = []

            for vae_start in range(0, total_frames, vae_chunk_size):
                vae_end = min(vae_start + vae_chunk_size, total_frames)
                print(f">>> Decoding frames {vae_start}-{vae_end}...", flush=True)

                chunk = latent_visual[vae_start:vae_end]
                images = chunk.reshape(
                    1,
                    -1,
                    chunk.shape[-3],
                    chunk.shape[-2],
                    chunk.shape[-1],
                )
                images = images.to(device=vae_device)
                images = (images / vae.config.scaling_factor).permute(0, 4, 1, 2, 3)
                images = vae.decode(images).sample
                images = ((images.clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8)

                decoded_chunks.append(images.cpu())
                torch.cuda.empty_cache()

            # Concatenate all decoded chunks along temporal dimension
            # Each chunk is [1, C, T, H, W]
            final_images = torch.cat(decoded_chunks, dim=2)

    log_vram_usage("AFTER VAE DECODE, BEFORE OFFLOAD (DENOISE)", dit=dit, vae=vae, text_embedder=None)

    if offload or force_offload:
        vae = vae.to('cpu', non_blocking=True)
    torch.cuda.empty_cache()

    return final_images


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
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if first_frames is not None:
                first_frames = first_frames.to(device=latent_visual.device, dtype=latent_visual.dtype)
                latent_visual[:1] = first_frames
                latent_visual = normalize_first_frame(latent_visual)

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


@torch.no_grad()
def generate_sample_t2v_pipeline_parallel(
    shape,
    caption,
    dit_gpu0,
    dit_gpu1,
    vae,
    conf,
    text_embedder,
    num_steps=50,
    guidance_weight=5.0,
    scheduler_scale=5.0,
    negative_caption="",
    clip_prompt=None,
    seed=6554,
    device0=torch.device('cuda:0'),
    device1=torch.device('cuda:1'),
    vae_device=None,
    progress=True,
):
    """
    Generate video using pipeline parallelism across two GPUs.

    GPU 0 runs diffusion steps 0 to N/2, then transfers latent to GPU 1.
    GPU 1 runs diffusion steps N/2 to N, then decodes with VAE.

    Args:
        shape: Output shape (bs, duration, height, width, dim)
        caption: Text prompt
        dit_gpu0: DiT model on GPU 0 (with block swapping)
        dit_gpu1: DiT model on GPU 1 (with block swapping)
        vae: VAE model (will be moved to vae_device for decode)
        conf: Model configuration
        text_embedder: Text encoder
        num_steps: Total diffusion steps
        guidance_weight: CFG weight
        scheduler_scale: Scheduler scale factor
        negative_caption: Negative prompt
        clip_prompt: Optional CLIP prompt
        seed: Random seed
        device0: First GPU device
        device1: Second GPU device
        vae_device: Device for VAE decode (defaults to device1)
        progress: Show progress bar

    Returns:
        Generated video tensor
    """
    if vae_device is None:
        vae_device = device1

    bs, duration, height, width, dim = shape
    mid_step = num_steps // 2

    print(f"\n{'='*60}")
    print(f"Pipeline Parallel Generation")
    print(f"  Total steps: {num_steps}")
    print(f"  GPU 0 steps: 0-{mid_step}")
    print(f"  GPU 1 steps: {mid_step}-{num_steps}")
    print(f"{'='*60}\n")

    # === Phase 1: Text Encoding (on GPU 0) ===
    print("[PIPELINE] Phase 1: Text encoding on GPU 0...", flush=True)
    t0 = time.perf_counter()
    text_embedder = text_embedder.to(device0)

    clip_texts = [clip_prompt] if clip_prompt else None
    bs_text_embed, text_cu_seqlens, attention_mask = text_embedder.encode(
        [caption], type_of_content="video", clip_texts=clip_texts
    )
    bs_null_text_embed, null_text_cu_seqlens, null_attention_mask = text_embedder.encode(
        [negative_caption], type_of_content="video"
    )
    print(f"[PIPELINE] Text encoding done in {time.perf_counter() - t0:.1f}s", flush=True)

    # Offload text encoder
    text_embedder = text_embedder.to('cpu')
    torch.cuda.empty_cache()

    # Prepare embeddings for both GPUs
    text_cu_seqlens_val = text_cu_seqlens.to(device=device0)[-1].item()
    null_text_cu_seqlens_val = null_text_cu_seqlens.to(device=device0)[-1].item()

    # GPU 0 embeddings
    bs_text_embed_gpu0 = {k: v.to(device=device0, dtype=torch.bfloat16) for k, v in bs_text_embed.items()}
    bs_null_text_embed_gpu0 = {k: v.to(device=device0, dtype=torch.bfloat16) for k, v in bs_null_text_embed.items()}
    attention_mask_gpu0 = attention_mask.to(device=device0)
    null_attention_mask_gpu0 = null_attention_mask.to(device=device0)

    # GPU 1 embeddings (copy)
    bs_text_embed_gpu1 = {k: v.to(device=device1, dtype=torch.bfloat16) for k, v in bs_text_embed.items()}
    bs_null_text_embed_gpu1 = {k: v.to(device=device1, dtype=torch.bfloat16) for k, v in bs_null_text_embed.items()}
    attention_mask_gpu1 = attention_mask.to(device=device1)
    null_attention_mask_gpu1 = null_attention_mask.to(device=device1)

    # RoPE positions
    visual_rope_pos = [
        torch.arange(duration),
        torch.arange(height // conf.model.dit_params.patch_size[1]),
        torch.arange(width // conf.model.dit_params.patch_size[2]),
    ]
    text_rope_pos = torch.arange(text_cu_seqlens_val)
    null_text_rope_pos = torch.arange(null_text_cu_seqlens_val)

    # === Phase 2: GPU 0 - First half of steps ===
    print(f"\n[PIPELINE] Phase 2: GPU 0 running steps 0-{mid_step}...", flush=True)
    t0 = time.perf_counter()

    # Move DiT GPU 0 non-block components to GPU (blocks stay on CPU for block swap)
    dit_gpu0.to(device0, non_blocking=True)

    # Ensure DiT GPU 0 is ready
    log_vram_usage("BEFORE GPU 0 DiT INFERENCE", dit=dit_gpu0, vae=None, text_embedder=None)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        intermediate_latent = generate(
            dit_gpu0,
            device0,
            (bs * duration, height, width, dim),
            num_steps,
            bs_text_embed_gpu0,
            bs_null_text_embed_gpu0,
            visual_rope_pos,
            text_rope_pos,
            null_text_rope_pos,
            guidance_weight,
            scheduler_scale,
            None,  # first_frames (T2V has no conditioning frames)
            conf,
            seed=seed,
            progress=progress,
            attention_mask=attention_mask_gpu0,
            null_attention_mask=null_attention_mask_gpu0,
            start_step=0,
            end_step=mid_step,
        )

    gpu0_time = time.perf_counter() - t0
    print(f"[PIPELINE] GPU 0 completed in {gpu0_time:.1f}s", flush=True)

    # Offload GPU 0 blocks to CPU
    if hasattr(dit_gpu0, 'offload_all_blocks'):
        dit_gpu0.offload_all_blocks()
    torch.cuda.empty_cache()

    # === Phase 3: Transfer latent to GPU 1 ===
    print(f"\n[PIPELINE] Phase 3: Transferring latent to GPU 1...", flush=True)
    t0 = time.perf_counter()
    intermediate_latent = intermediate_latent.to(device1)
    transfer_time = time.perf_counter() - t0
    print(f"[PIPELINE] Transfer completed in {transfer_time*1000:.1f}ms", flush=True)

    # === Phase 4: GPU 1 - Second half of steps ===
    print(f"\n[PIPELINE] Phase 4: GPU 1 running steps {mid_step}-{num_steps}...", flush=True)
    t0 = time.perf_counter()

    # Move DiT GPU 1 non-block components to GPU (blocks stay on CPU for block swap)
    dit_gpu1.to(device1, non_blocking=True)

    log_vram_usage("BEFORE GPU 1 DiT INFERENCE", dit=dit_gpu1, vae=None, text_embedder=None)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        final_latent = generate(
            dit_gpu1,
            device1,
            (bs * duration, height, width, dim),
            num_steps,
            bs_text_embed_gpu1,
            bs_null_text_embed_gpu1,
            visual_rope_pos,
            text_rope_pos,
            null_text_rope_pos,
            guidance_weight,
            scheduler_scale,
            None,  # first_frames
            conf,
            seed=seed,
            progress=progress,
            attention_mask=attention_mask_gpu1,
            null_attention_mask=null_attention_mask_gpu1,
            start_step=mid_step,
            end_step=num_steps,
            initial_latent=intermediate_latent,
        )

    gpu1_time = time.perf_counter() - t0
    print(f"[PIPELINE] GPU 1 completed in {gpu1_time:.1f}s", flush=True)

    # Offload GPU 1 blocks
    if hasattr(dit_gpu1, 'offload_all_blocks'):
        dit_gpu1.offload_all_blocks()
    torch.cuda.empty_cache()

    # === Phase 5: VAE Decode ===
    print(f"\n[PIPELINE] Phase 5: VAE decoding on {vae_device}...", flush=True)
    t0 = time.perf_counter()

    vae = vae.to(vae_device)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        images = final_latent.reshape(
            bs,
            -1,
            final_latent.shape[-3],
            final_latent.shape[-2],
            final_latent.shape[-1],
        )
        images = images.to(device=vae_device)
        images = (images / vae.config.scaling_factor).permute(0, 4, 1, 2, 3)
        images = vae.decode(images).sample
        images = ((images.clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8)

    vae_time = time.perf_counter() - t0
    print(f"[PIPELINE] VAE decode completed in {vae_time:.1f}s", flush=True)

    # Offload VAE
    vae = vae.to('cpu')
    torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print(f"Pipeline Parallel Generation Complete!")
    print(f"  GPU 0 time: {gpu0_time:.1f}s")
    print(f"  Transfer time: {transfer_time*1000:.1f}ms")
    print(f"  GPU 1 time: {gpu1_time:.1f}s")
    print(f"  VAE time: {vae_time:.1f}s")
    print(f"  Total DiT time: {gpu0_time + gpu1_time:.1f}s")
    print(f"{'='*60}\n")

    return images
