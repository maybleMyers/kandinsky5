from typing import Union, Optional

import transformers
import torch
from torchvision.transforms import ToPILImage
from .generation_utils import generate_sample

torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose = True

class Kandinsky5T2IPipeline:
    def __init__(
        self,
        device_map: Union[
            str, torch.device, dict
        ],  # {"dit": cuda:0, "vae": cuda:1, "text_embedder": cuda:1 }
        dit,
        text_embedder,
        vae,
        resolution: int = 1024,
        local_dit_rank: int = 0,
        world_size: int = 1,
        conf = None,
        offload: bool = False,
    ):
        if resolution not in [1024]:
            raise ValueError("Resolution can be only 1024") 

        self.dit = dit
        self.text_embedder = text_embedder
        self.vae = vae

        self.resolution = resolution

        self.device_map = device_map
        self.local_dit_rank = local_dit_rank
        self.world_size = world_size
        self.conf = conf
        self.num_steps = conf.model.num_steps
        self.guidance_weight = conf.model.guidance_weight

        self.offload = offload

        self.RESOLUTIONS = {
            1024: [(1024, 1024), (640, 1408), (1408, 640), (768, 1280), (1280, 768), (896, 1152), (1152, 896)],
        }

    def expand_prompt(self, prompt):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""Rewrite and enhance the original prompt with richer detail, clearer structure, and improved descriptive quality. Expand the scene, atmosphere, and context while preserving the user’s intent. When adding text that should appear inside an image, place that text inside double quotes and in capital letters. Strengthen visual clarity, style, and specificity, but do not change the meaning. Output only the enhanced prompt, written in polished, vivid language suitable for high-quality image generation.
        example:
        Original text: white mini police car with blue stripes, with 911 and 'Police' text 
        Result: A miniature model car simulating the official transport of the relevant authorities. The body is white with blue stripes. The word "911" is written in large blue letters on the hood and side. Below it, "POLICE" is used in a font. The windows are transparent, and the interior has black seats. The headlights have plastic lenses, and the roof has blue and red beacons. The radiator grille has vertical slots. The wheels are black with white rims. The doors are closed, the windows have black frames. The background is uniform white.
        Here 911 in double quotes because it is text on image, 'Police' -> "POLICE" because it should be in double quotes and capital letters.
        Rewrite Prompt: "{prompt}". Answer only with expanded prompt.""",
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
        inputs = inputs.to(self.text_embedder.embedder.model.device)
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
        width: int = 1024,
        height: int = 1024,
        seed: int = None,
        num_steps: Optional[int] = None,
        guidance_weight: Optional[float] = None,
        scheduler_scale: float = 3.0,
        negative_caption: str = "",
        expand_prompts: bool = True,
        save_path: str = None,
        progress: bool = True,
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

        if self.resolution != 1024:
            raise NotImplementedError("Only 1024 resolution is available for now")

        if (height, width) not in self.RESOLUTIONS[self.resolution]:
            raise ValueError(
                f"Wrong height, width pair. Available (height, width) are: {self.RESOLUTIONS[self.resolution]}"
            )

        caption = text
        if expand_prompts:
            transformers.set_seed(seed)
            if self.local_dit_rank == 0:
                if self.offload:
                    self.text_embedder = self.text_embedder.to(self.device_map["text_embedder"])
                caption = self.expand_prompt(caption)
            if self.world_size > 1:
                caption = [caption]
                torch.distributed.broadcast_object_list(caption, 0)
                caption = caption[0]

        shape = (1, 1, height // 8, width // 8, 16)

        # GENERATION
        images = generate_sample(
            shape,
            caption,
            self.dit,
            self.vae,
            self.conf,
            text_embedder=self.text_embedder,
            num_steps=num_steps,
            guidance_weight=guidance_weight,
            scheduler_scale=scheduler_scale,
            negative_caption=negative_caption,
            seed=seed,
            device=self.device_map["dit"],
            vae_device=self.device_map["vae"],
            text_embedder_device=self.device_map["text_embedder"],
            progress=progress,
            offload=self.offload,
            image_vae=True
        )
        torch.cuda.empty_cache()

        # RESULTS
        if self.local_dit_rank == 0:
            return_images = []
            for image in images.cpu():
                return_images.append(ToPILImage()(image))
            if save_path is not None:
                if isinstance(save_path, str):
                    save_path = [save_path]
                if len(save_path) == len(return_images):
                    for path, image in zip(save_path, return_images):
                        image.save(path)
            return return_images