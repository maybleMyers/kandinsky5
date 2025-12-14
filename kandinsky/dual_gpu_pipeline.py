"""
Dual GPU Pipeline for Kandinsky 5 with Block Swapping.

Enables pipeline parallelism across two GPUs where each GPU has its own model copy
with independent block swapping to CPU. This allows generating video faster by
splitting diffusion steps between GPUs.
"""

from typing import Union
import gc

import transformers
import torch
import torchvision
from torchvision.transforms import ToPILImage

from .generation_utils import generate_sample_t2v_pipeline_parallel

torch._dynamo.config.suppress_errors = True


class Kandinsky5DualGPUPipeline:
    """
    Dual GPU pipeline for Kandinsky 5 video generation.

    Each GPU has its own DiT model with independent block swapping.
    Steps are split between GPUs for pipeline parallelism.
    """

    def __init__(
        self,
        dit_gpu0,           # DiT model for GPU 0 (with block swapping)
        dit_gpu1,           # DiT model for GPU 1 (with block swapping)
        vae,
        text_embedder,
        conf,
        device0: torch.device = torch.device('cuda:0'),
        device1: torch.device = torch.device('cuda:1'),
        resolution: int = 512,
    ):
        """
        Initialize dual GPU pipeline.

        Args:
            dit_gpu0: DiT model for GPU 0 (already loaded with block swap)
            dit_gpu1: DiT model for GPU 1 (already loaded with block swap)
            vae: VAE model (will be moved to appropriate device for decode)
            text_embedder: Text encoder (will be moved between devices)
            conf: Model configuration
            device0: First GPU device
            device1: Second GPU device
            resolution: Target resolution
        """
        self.dit_gpu0 = dit_gpu0
        self.dit_gpu1 = dit_gpu1
        self.vae = vae
        self.text_embedder = text_embedder
        self.conf = conf
        self.device0 = device0
        self.device1 = device1
        self.resolution = resolution

        self.num_steps = conf.model.num_steps
        self.guidance_weight = conf.model.guidance_weight

        self.RESOLUTIONS = {
            512: [(512, 512), (512, 768), (768, 512)],
            1024: [(1024, 1024), (1280, 768), (768, 1280), (1408, 640), (640, 1408), (1152, 896), (896, 1152)],
        }

    def expand_prompt(self, prompt, device):
        """Expand prompt using Qwen model."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""You are a prompt beautifier that transforms short user video descriptions into rich, detailed English prompts specifically optimized for video generation models.
        Here are some example descriptions from the dataset that the model was trained:
        1. "In a dimly lit room with a cluttered background, papers are pinned to the wall and various objects rest on a desk. Three men stand present: one wearing a red sweater, another in a black sweater, and the third in a gray shirt. The man in the gray shirt speaks and makes hand gestures, while the other two men look forward. The camera remains stationary, focusing on the three men throughout the sequence. A gritty and realistic visual style prevails, marked by a greenish tint that contributes to a moody atmosphere. Low lighting casts shadows, enhancing the tense mood of the scene."
        2. "In an office setting, a man sits at a desk wearing a gray sweater and seated in a black office chair. A wooden cabinet with framed pictures stands beside him, alongside a small plant and a lit desk lamp. Engaged in a conversation, he makes various hand gestures to emphasize his points. His hands move in different positions, indicating different ideas or points. The camera remains stationary, focusing on the man throughout. Warm lighting creates a cozy atmosphere. The man appears to be explaining something. The overall visual style is professional and polished, suitable for a business or educational context."
        3. "A person works on a wooden object resembling a sunburst pattern, holding it in their left hand while using their right hand to insert a thin wire into the gaps between the wooden pieces. The background features a natural outdoor setting with greenery and a tree trunk visible. The camera stays focused on the hands and the wooden object throughout, capturing the detailed process of assembling the wooden structure. The person carefully threads the wire through the gaps, ensuring the wooden pieces are securely fastened together. The scene unfolds with a naturalistic and instructional style, emphasizing the craftsmanship and the methodical steps taken to complete the task."
        Importantly! These are just examples from a large training dataset of 200 million videos.
        Rewrite Prompt: "{prompt}" to get high-quality video generation. Answer only with expanded prompt.""",
                    },
                ],
            }
        ]
        text = self.text_embedder.embedder.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.text_embedder.embedder.processor(
            text=[text],
            images=None,
            videos=None,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)
        generated_ids = self.text_embedder.embedder.model.generate(
            **inputs, max_new_tokens=256
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.text_embedder.embedder.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0]

    def __call__(
        self,
        text: str,
        time_length: int = 5,
        width: int = 768,
        height: int = 512,
        seed: int = None,
        num_steps: int = None,
        guidance_weight: float = None,
        scheduler_scale: float = 10.0,
        negative_caption: str = "Static, 2D cartoon, cartoon, 2d animation, paintings, images, worst quality, low quality, ugly, deformed, walking backwards",
        expand_prompts: bool = True,
        clip_prompt: str = None,
        save_path: str = None,
        progress: bool = True,
    ):
        """
        Generate video using pipeline parallelism across two GPUs.

        Args:
            text: Text prompt for video generation
            time_length: Video duration in seconds
            width: Video width in pixels
            height: Video height in pixels
            seed: Random seed
            num_steps: Number of diffusion steps
            guidance_weight: Classifier-free guidance weight
            scheduler_scale: Scheduler scale factor
            negative_caption: Negative prompt
            expand_prompts: Whether to expand prompts using Qwen
            clip_prompt: Optional CLIP prompt
            save_path: Path to save output video
            progress: Show progress bar

        Returns:
            Generated video tensor or PIL images
        """
        num_steps = self.num_steps if num_steps is None else num_steps
        guidance_weight = self.guidance_weight if guidance_weight is None else guidance_weight

        # Seed
        if seed is None:
            seed = torch.randint(2**32 - 1, (1,)).item()

        # Preparation
        num_frames = 1 if time_length == 0 else time_length * 24 // 4 + 1

        caption = text
        if expand_prompts:
            transformers.set_seed(seed)
            # Move text embedder to GPU 0 for prompt expansion
            self.text_embedder = self.text_embedder.to(self.device0)
            caption = self.expand_prompt(caption, self.device0)
            print("\n" + "="*80)
            print("EXPANDED QWEN 2.5 PROMPT:")
            print("="*80)
            print(caption)
            print("="*80 + "\n")
            # Offload after expansion
            self.text_embedder = self.text_embedder.to('cpu')
            torch.cuda.empty_cache()

        shape = (1, num_frames, height // 8, width // 8, 16)

        # Generate using pipeline parallelism
        images = generate_sample_t2v_pipeline_parallel(
            shape,
            caption,
            self.dit_gpu0,
            self.dit_gpu1,
            self.vae,
            self.conf,
            text_embedder=self.text_embedder,
            num_steps=num_steps,
            guidance_weight=guidance_weight,
            scheduler_scale=scheduler_scale,
            negative_caption=negative_caption,
            clip_prompt=clip_prompt,
            seed=seed,
            device0=self.device0,
            device1=self.device1,
            vae_device=self.device1,  # Decode on GPU 1
            progress=progress,
        )

        # Results
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
                for i, video in enumerate(images.permute(0, 2, 1, 3, 4)):
                    torchvision.io.write_video(
                        save_path[i] if i < len(save_path) else f"output_{i}.mp4",
                        video.cpu(),
                        fps=24,
                        video_codec="h264",
                        options={"crf": "18"},
                    )
            return images.permute(0, 2, 1, 3, 4)
