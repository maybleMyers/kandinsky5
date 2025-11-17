import torch
from comfy_api.latest import io
import comfy.utils
import comfy.model_management

from comfy.cli_args import args
from PIL import Image
import torch.nn.functional as Fmod

from .src.kandinsky.magcache_utils import set_magcache_params, disable_magcache
from .src.kandinsky.models.utils import fast_sta_nabla
from .src.kandinsky.models.nn import set_sage_attention

# --- VHS / latent preview utils ---

import server
from threading import Thread
import io as pyio
import time
import struct
from importlib.util import find_spec

serv = server.PromptServer.instance

MAX_PREVIEW_RESOLUTION = args.preview_size

# Współczynniki z HyVideoLatentPreview (Hunyuan VAE)
HYVIDEO_LATENT_RGB_FACTORS = [
    [-0.41, -0.25, -0.26],
    [-0.26, -0.49, -0.24],
    [-0.37, -0.54, -0.3],
    [-0.04, -0.29, -0.29],
    [-0.52, -0.59, -0.39],
    [-0.56, -0.6, -0.02],
    [-0.53, -0.06, -0.48],
    [-0.51, -0.28, -0.18],
    [-0.59, -0.1, -0.33],
    [-0.56, -0.54, -0.41],
    [-0.61, -0.19, -0.5],
    [-0.05, -0.25, -0.17],
    [-0.23, -0.04, -0.22],
    [-0.51, -0.56, -0.43],
    [-0.13, -0.4, -0.05],
    [-0.01, -0.01, -0.48],
]
HYVIDEO_LATENT_RGB_BIAS = [0.0, 0.0, 0.0]


class Latent2RGBPreviewer:
    def __init__(self, latent_rgb_factors, latent_rgb_factors_bias=None):
        self.latent_rgb_factors = torch.tensor(latent_rgb_factors, device="cpu").transpose(0, 1)
        self.latent_rgb_factors_bias = None
        if latent_rgb_factors_bias is not None:
            self.latent_rgb_factors_bias = torch.tensor(latent_rgb_factors_bias, device="cpu")

    def decode_latent_to_preview(self, x0):
        # x0: (N, C, H, W)
        self.latent_rgb_factors = self.latent_rgb_factors.to(dtype=x0.dtype, device=x0.device)
        if self.latent_rgb_factors_bias is not None:
            self.latent_rgb_factors_bias = self.latent_rgb_factors_bias.to(dtype=x0.dtype, device=x0.device)

        # (N, C, H, W) -> (N, H, W, C)
        latent_image = Fmod.linear(
            x0.movedim(1, -1),
            self.latent_rgb_factors,
            bias=self.latent_rgb_factors_bias,
        )  # (N, H, W, 3)

        # normalizacja min/max jak w HyVideoLatentPreview
        img_min = latent_image.min()
        img_max = latent_image.max()
        if (img_max - img_min) > 1e-6:
            latent_image = (latent_image - img_min) / (img_max - img_min)
        else:
            latent_image = torch.zeros_like(latent_image)

        return latent_image


class WrappedPreviewer:
    """
    Uproszczona wersja z VideoHelperSuite:
    - przyjmuje video latent x0 shape (1, C, F, H, W)
    - streamuje animację do klienta przez PREVIEW_IMAGE
    """

    def __init__(self, previewer, rate=16):
        self.first_preview = True
        self.last_time = 0
        self.c_index = 0
        self.rate = rate
        self.swarmui_env = find_spec("SwarmComfyCommon") is not None
        if self.swarmui_env:
            print("previewer: SwarmUI output enabled")

        # używamy tylko latent2rgb jako źródła współczynników
        if hasattr(previewer, "latent_rgb_factors"):
            self.latent_rgb_factors = previewer.latent_rgb_factors
            self.latent_rgb_factors_bias = previewer.latent_rgb_factors_bias
        else:
            raise Exception("Unsupported preview type for VHS animated previews")

    def decode_latent_to_preview(self, x0):
        # x0: (N, C, H, W)
        self.latent_rgb_factors = self.latent_rgb_factors.to(dtype=x0.dtype, device=x0.device)
        if self.latent_rgb_factors_bias is not None:
            self.latent_rgb_factors_bias = self.latent_rgb_factors_bias.to(
                dtype=x0.dtype, device=x0.device
            )

        latent_image = Fmod.linear(
            x0.movedim(1, -1),
            self.latent_rgb_factors,
            bias=self.latent_rgb_factors_bias,
        )  # (N, H, W, 3)

        # min/max jak w HyVideoLatentPreview
        img_min = latent_image.min()
        img_max = latent_image.max()
        if (img_max - img_min) > 1e-6:
            latent_image = (latent_image - img_min) / (img_max - img_min)
        else:
            latent_image = torch.zeros_like(latent_image)

        return latent_image

    def decode_latent_to_preview_image(self, preview_format, x0):
        """
        x0:
          - (1, C, F, H, W) - batch video
          - lub (N, C, H, W) - pojedyncze klatki
        """
        if x0.ndim == 5:
            # (B, C, F, H, W) -> (F*B, C, H, W)
            x0 = x0.movedim(2, 1)
            x0 = x0.reshape((-1,) + x0.shape[-3:])

        num_images = x0.size(0)
        new_time = time.time()
        num_previews = int((new_time - self.last_time) * self.rate)
        self.last_time = self.last_time + num_previews / self.rate

        if num_previews > num_images:
            num_previews = num_images
        elif num_previews <= 0:
            return None

        if self.first_preview:
            self.first_preview = False
            serv.send_sync(
                "VHS_latentpreview",
                {"length": num_images, "rate": self.rate, "id": serv.last_node_id},
            )
            self.last_time = new_time + 1 / self.rate

        if self.c_index + num_previews > num_images:
            x0 = x0.roll(-self.c_index, 0)[:num_previews]
        else:
            x0 = x0[self.c_index : self.c_index + num_previews]

        Thread(
            target=self.process_previews,
            args=(x0, self.c_index, num_images),
        ).start()

        self.c_index = (self.c_index + num_previews) % num_images
        return None

    def process_previews(self, image_tensor, ind, leng):
        max_size = 256

        # image_tensor: (N, C, H, W)
        image_tensor = self.decode_latent_to_preview(image_tensor)

        # teraz (N, H, W, 3), skaluje do max_size
        if image_tensor.size(1) > max_size or image_tensor.size(2) > max_size:
            image_tensor = image_tensor.movedim(-1, 0)
            if image_tensor.size(2) < image_tensor.size(3):
                height = (max_size * image_tensor.size(2)) // image_tensor.size(3)
                image_tensor = Fmod.interpolate(image_tensor, (height, max_size), mode="bilinear")
            else:
                width = (max_size * image_tensor.size(3)) // image_tensor.size(2)
                image_tensor = Fmod.interpolate(image_tensor, (max_size, width), mode="bilinear")
            image_tensor = image_tensor.movedim(0, -1)

        previews_ubyte = (
            image_tensor.clamp(0, 1)
            .mul(0xFF)
            .to(device="cpu", dtype=torch.uint8)
        )

        # klasyczny VHS PREVIEW_IMAGE po jednej klatce
        for preview in previews_ubyte:
            img = Image.fromarray(preview.numpy())
            message = pyio.BytesIO()
            message.write((1).to_bytes(length=4, byteorder="big") * 2)
            message.write(ind.to_bytes(length=4, byteorder="big"))
            message.write(struct.pack("16p", serv.last_node_id.encode("ascii")))
            img.save(message, format="JPEG", quality=95, compress_level=1)
            serv.send_sync(server.BinaryEventTypes.PREVIEW_IMAGE, message.getvalue(), serv.client_id)
            ind = (ind + 1) % leng

        # SwarmUI multi-frame WEBP jeśli jest
        if self.swarmui_env:
            images = [Image.fromarray(preview.numpy()) for preview in previews_ubyte]
            message = pyio.BytesIO()
            header = struct.pack(">I", 3)
            message.write(header)
            images[0].save(
                message,
                save_all=True,
                duration=int(1000.0 / self.rate),
                append_images=images[1:],
                lossless=False,
                quality=80,
                method=0,
                format="WEBP",
            )
            message.seek(0)
            preview_bytes = message.getvalue()
            serv.send_sync(1, preview_bytes, sid=serv.client_id)


def get_kandinsky_video_previewer():
    """Buduje WrappedPreviewer z hunyuanowymi latent_rgb_factors."""
    previewer_core = Latent2RGBPreviewer(HYVIDEO_LATENT_RGB_FACTORS, HYVIDEO_LATENT_RGB_BIAS)
    return WrappedPreviewer(previewer_core, rate=16)


# --- Kandinsky sampler ---

@torch.no_grad()
def get_sparse_params(conf, batch_shape, device):
    F_cond, H_cond, W_cond, C_cond = batch_shape
    patch_size = conf.model.dit_params.patch_size

    T = F_cond // patch_size[0]
    H = H_cond // patch_size[1]
    W = W_cond // patch_size[2]

    if conf.model.attention.type == "nabla":
        sta_mask = fast_sta_nabla(
            T,
            H // 8,
            W // 8,
            conf.model.attention.wT,
            conf.model.attention.wH,
            conf.model.attention.wW,
            device=device,
        )
        sparse_params = {
            "sta_mask": sta_mask.unsqueeze_(0).unsqueeze_(0),
            "attention_type": conf.model.attention.type,
            "to_fractal": True,
            "P": conf.model.attention.P,
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
    visual_cond=None,
    visual_cond_mask=None,
):
    model_input = x
    if dit.visual_cond:
        if visual_cond is not None and visual_cond_mask is not None:
            visual_cond_input = torch.zeros_like(x)
            visual_cond_input[0:1] = visual_cond.to(dtype=x.dtype)

            visual_cond_mask_input = visual_cond_mask.to(dtype=x.dtype)
        else:
            visual_cond_input = torch.zeros_like(x)
            visual_cond_mask_input = torch.zeros([*x.shape[:-1], 1], dtype=x.dtype, device=x.device)

        model_input = torch.cat([x, visual_cond_input, visual_cond_mask_input], dim=-1)

    pred_velocity = dit(
        model_input,
        text_embeds["text_embeds"],
        text_embeds["pooled_embed"],
        t * 1000,
        visual_rope_pos,
        text_rope_pos,
        scale_factor=conf.metrics.scale_factor,
        sparse_params=sparse_params,
    )

    if abs(guidance_weight - 1.0) > 1e-6:
        uncond_pred_velocity = dit(
            model_input,
            null_text_embeds["text_embeds"],
            null_text_embeds["pooled_embed"],
            t * 1000,
            visual_rope_pos,
            null_text_rope_pos,
            scale_factor=conf.metrics.scale_factor,
            sparse_params=sparse_params,
        )
        pred_velocity = torch.lerp(uncond_pred_velocity, pred_velocity, guidance_weight)

    return pred_velocity


@torch.no_grad()
def generate(
    diffusion_model,
    device,
    shape,
    steps,
    text_embed,
    null_embed,
    visual_rope_pos,
    text_rope_pos,
    null_text_rope_pos,
    cfg,
    scheduler_scale,
    conf,
    seed,
    pbar,
    visual_cond=None,
    visual_cond_mask=None,
    global_step_offset=0,
    total_steps=None,
    video_previewer=None,
):
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    model_dtype = next(diffusion_model.parameters()).dtype
    current_latent = torch.randn(shape, generator=g, device=device, dtype=model_dtype)

    if torch.isnan(current_latent).any() or torch.isinf(current_latent).any():
        current_latent = torch.randn(shape, device=device, dtype=model_dtype)

    lock_first_frame = False
    if visual_cond is not None and visual_cond_mask is not None:
        if visual_cond_mask[0].sum() > 0:
            visual_cond_typed = visual_cond.to(dtype=model_dtype)
            current_latent[0:1] = visual_cond_typed
            lock_first_frame = True

    sparse_params = get_sparse_params(conf, shape, device)

    timesteps = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=model_dtype)
    timesteps = scheduler_scale * timesteps / (1 + (scheduler_scale - 1) * timesteps)

    for i in range(steps):
        t_now = timesteps[i]
        t_next = timesteps[i + 1]
        dt = t_next - t_now

        pred_velocity = get_velocity(
            diffusion_model,
            current_latent,
            t_now.unsqueeze(0),
            text_embed,
            null_embed,
            visual_rope_pos,
            text_rope_pos,
            null_text_rope_pos,
            cfg,
            conf,
            sparse_params=sparse_params,
            visual_cond=visual_cond,
            visual_cond_mask=visual_cond_mask,
        )

        if torch.isnan(pred_velocity).any() or torch.isinf(pred_velocity).any():
            pred_velocity = torch.nan_to_num(pred_velocity, nan=0.0, posinf=0.0, neginf=0.0)

        if lock_first_frame:
            pred_velocity[0:1] = 0.0

        # przybliżenie x0 dla podglądu
        try:
            t_scalar = float(t_now)
        except Exception:
            t_scalar = 0.0
        x0_approx = current_latent - t_scalar * pred_velocity
        x0_approx = torch.nan_to_num(x0_approx, nan=0.0, posinf=0.0, neginf=0.0)

        # video preview co step: całe video (F, H, W, C)
        if video_previewer is not None:
            try:
                # (F, H, W, C) -> (1, C, F, H, W)
                x0_video = x0_approx.permute(3, 0, 1, 2).unsqueeze(0)
                video_previewer.decode_latent_to_preview_image("JPEG", x0_video)
            except Exception as e:
                print("Kandinsky video preview error:", e)
                video_previewer = None

        # update stanu latentu
        current_latent = current_latent + dt * pred_velocity

        max_val = current_latent.abs().max()
        if max_val > 50.0:
            current_latent = current_latent * (50.0 / max_val)

        current_latent = torch.nan_to_num(current_latent, nan=0.0, posinf=0.0, neginf=0.0)

        # progress bar co step
        global_step = global_step_offset + i
        if hasattr(pbar, "update_absolute") and total_steps is not None:
            pbar.update_absolute(global_step + 1, total_steps, None)
        else:
            pbar.update(1)

    return current_latent


class KandinskySampler(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="KandinskyV5_Sampler",
            display_name="Kandinsky 5 Sampler",
            category="Kandinsky",
            description="Performs the specific Flow Matching sampling loop for Kandinsky-5 models.",
            inputs=[
                io.Model.Input("model", tooltip="The Kandinsky 5 model patcher from the Kandinsky 5 Loader."),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True),
                io.Int.Input("steps", default=50, min=1, max=200, tooltip="50, 16 for distilled version."),
                io.Float.Input(
                    "cfg",
                    default=5.0,
                    min=1.0,
                    max=20.0,
                    step=0.1,
                    tooltip="1.0 for distilled16steps and nocfg, 5.0 for sft and pretrain.",
                ),
                io.Float.Input(
                    "scheduler_scale",
                    default=1.0,
                    min=1.0,
                    max=20.0,
                    step=0.1,
                    tooltip="5.0 for 5s, 10.0 for 10s.",
                ),
                io.Boolean.Input(
                    "use_sage_attention",
                    default=False,
                    tooltip="Enable SageAttention for faster inference with lower memory usage.",
                ),
                io.Conditioning.Input("positive", tooltip="Positive conditioning from Kandinsky 5 Text Encode."),
                io.Conditioning.Input("negative", tooltip="Negative conditioning from Kandinsky 5 Text Encode."),
                io.Latent.Input("latent_image", tooltip="Empty latent from Empty Kandinsky 5 Latent."),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    @torch.no_grad()
    def execute(
        cls,
        model,
        seed,
        steps,
        cfg,
        scheduler_scale,
        use_sage_attention,
        positive,
        negative,
        latent_image,
    ) -> io.NodeOutput:
        patcher = model
        set_sage_attention(use_sage_attention)

        comfy.model_management.load_model_gpu(patcher)
        k_handler = patcher.model
        diffusion_model = k_handler.diffusion_model
        conf = k_handler.conf
        device = patcher.load_device
        model_dtype = next(diffusion_model.parameters()).dtype

        use_magcache = conf.get("use_magcache", False)
        is_magcache_active = hasattr(diffusion_model, "_magcache_enabled") and diffusion_model._magcache_enabled

        if use_magcache:
            if hasattr(conf, "magcache"):
                threshold = conf.get("magcache_threshold", 0.12)

                set_magcache_params(
                    diffusion_model,
                    conf.magcache.mag_ratios,
                    steps,
                    conf.model.guidance_weight == 1.0,
                    threshold=threshold,
                    start_percent=0.2,
                    end_percent=1.0,
                )
            else:
                print("Warning: use_magcache is True but no magcache config found")
        elif is_magcache_active:
            disable_magcache(diffusion_model)

        latent = latent_image["samples"].to(device)
        B, C, F_frames, H, W = latent.shape

        visual_cond = None
        visual_cond_mask = None
        if "visual_cond" in latent_image and "visual_cond_mask" in latent_image:
            visual_cond = latent_image["visual_cond"].to(device=device, dtype=model_dtype)
            visual_cond_mask = latent_image["visual_cond_mask"].to(device=device, dtype=model_dtype)

        pos_cond = positive[0][1].get("kandinsky_embeds")
        neg_cond = negative[0][1].get("kandinsky_embeds")

        for key in pos_cond:
            pos_cond[key] = pos_cond[key].to(device=device, dtype=model_dtype)
            neg_cond[key] = neg_cond[key].to(device=device, dtype=model_dtype)

        patch_size = conf.model.dit_params.patch_size
        visual_rope_pos = [
            torch.arange(F_frames // patch_size[0], device=device),
            torch.arange(H // patch_size[1], device=device),
            torch.arange(W // patch_size[2], device=device),
        ]

        text_rope_pos = torch.arange(pos_cond["text_embeds"].shape[0], device=device)
        null_text_rope_pos = torch.arange(neg_cond["text_embeds"].shape[0], device=device)

        output_latents = []
        total_steps = steps * B
        pbar = comfy.utils.ProgressBar(total_steps)

        try:
            video_previewer = get_kandinsky_video_previewer()
        except Exception as e:
            print("Could not init Kandinsky video previewer:", e)
            video_previewer = None

        for i in range(B):
            current_seed = seed + i

            batch_visual_cond = None
            batch_visual_cond_mask = None
            if visual_cond is not None and visual_cond_mask is not None:
                batch_visual_cond = visual_cond[i].permute(1, 2, 3, 0)
                batch_visual_cond_mask = visual_cond_mask[i]

            final_latent_unbatched = generate(
                diffusion_model,
                device,
                (F_frames, H, W, C),
                steps,
                pos_cond,
                neg_cond,
                visual_rope_pos,
                text_rope_pos,
                null_text_rope_pos,
                cfg,
                scheduler_scale,
                conf,
                current_seed,
                pbar,
                visual_cond=batch_visual_cond,
                visual_cond_mask=batch_visual_cond_mask,
                global_step_offset=i * steps,
                total_steps=total_steps,
                video_previewer=video_previewer,
            )
            output_latents.append(final_latent_unbatched.permute(3, 0, 1, 2))

        final_latents = torch.stack(output_latents, dim=0)

        if torch.isnan(final_latents).any() or torch.isinf(final_latents).any():
            final_latents = torch.nan_to_num(final_latents, nan=0.0, posinf=0.0, neginf=0.0)

        scaling_factor = 0.476986
        scaled_latents = final_latents / scaling_factor

        if torch.isnan(scaled_latents).any() or torch.isinf(scaled_latents).any():
            scaled_latents = torch.nan_to_num(scaled_latents, nan=0.0, posinf=0.0, neginf=0.0)

        return io.NodeOutput({"samples": scaled_latents.to(comfy.model_management.intermediate_device())})
