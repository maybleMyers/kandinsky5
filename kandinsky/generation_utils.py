import os
os.environ["TOKENIZERS_PARALLELISM"] = "False"

import torch
from tqdm import tqdm

from .models.utils import fast_sta_nabla


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
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if first_frames is not None:
                first_frames = first_frames.to(device=latent_visual.device, dtype=latent_visual.dtype)
                latent_visual[:1] = first_frames
                latent_visual = normalize_first_frame(latent_visual)

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
