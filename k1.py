import gradio as gr
import os
import sys
import time
import random
import warnings
import logging
from typing import Generator, List, Tuple, Optional
import torch

from kandinsky import get_T2V_pipeline, get_I2V_pipeline, get_I2V_pipeline_with_block_swap

global_pipe = None
stop_event = threading.Event() if 'threading' in dir() else None

import threading
stop_event = threading.Event()

def disable_warnings():
    warnings.filterwarnings("ignore")
    logging.getLogger("torch").setLevel(logging.ERROR)
    torch._logging.set_logs(
        dynamo=logging.ERROR,
        dynamic=logging.ERROR,
        aot=logging.ERROR,
        inductor=logging.ERROR,
        guards=False,
        recompiles=False
    )

disable_warnings()

def load_pipeline(mode: str, enable_block_swap: bool, blocks_in_memory: int, dtype_str: str):
    global global_pipe

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    model_dtype = dtype_map[dtype_str]

    config_path = "./configs/config_5s_i2v_pro_20b.yaml"

    device_map = {"dit": "cuda:0", "vae": "cuda:0", "text_embedder": "cuda:0"}

    if mode == "i2v":
        if enable_block_swap:
            global_pipe = get_I2V_pipeline_with_block_swap(
                device_map=device_map,
                conf_path=config_path,
                offload=False,
                magcache=False,
                quantized_qwen=False,
                attention_engine="auto",
                blocks_in_memory=blocks_in_memory,
                enable_block_swap=True,
                dtype=model_dtype,
            )
        else:
            global_pipe = get_I2V_pipeline(
                device_map=device_map,
                conf_path=config_path,
                offload=False,
                magcache=False,
                quantized_qwen=False,
                attention_engine="auto",
                dtype=model_dtype,
            )
    else:
        global_pipe = get_T2V_pipeline(
            device_map=device_map,
            conf_path=config_path,
            offload=False,
            magcache=False,
            quantized_qwen=False,
            attention_engine="auto",
            dtype=model_dtype,
        )

    return "Model loaded successfully!"

def generate_video(
    prompt: str,
    negative_prompt: str,
    input_image: str,
    mode: str,
    width: int,
    height: int,
    video_duration: int,
    sample_steps: int,
    guidance_weight: float,
    scheduler_scale: float,
    seed: int,
    enable_block_swap: bool,
    blocks_in_memory: int,
    dtype_str: str,
    save_path: str,
    batch_size: int,
) -> Generator[Tuple[List[Tuple[str, str]], str, str], None, None]:
    global global_pipe, stop_event
    stop_event.clear()

    os.makedirs(save_path, exist_ok=True)
    all_generated_videos = []

    for i in range(int(batch_size)):
        if stop_event.is_set():
            yield all_generated_videos, "Generation stopped by user.", ""
            return

        current_seed = seed
        if seed == -1:
            current_seed = random.randint(0, 2**32 - 1)
        elif int(batch_size) > 1:
            current_seed = seed + i

        status_text = f"Processing {i+1}/{batch_size} (Seed: {current_seed})"
        yield all_generated_videos.copy(), status_text, "Starting generation..."

        if global_pipe is None:
            yield all_generated_videos, "Error: Model not loaded. Please load the model first.", ""
            return

        timestamp = int(time.time())
        output_filename = os.path.join(save_path, f"k1_{mode}_{timestamp}_{current_seed}.mp4")

        try:
            start_time = time.perf_counter()

            if mode == "i2v":
                if not input_image:
                    yield all_generated_videos, "Error: Input image required for i2v mode.", ""
                    return

                global_pipe(
                    prompt,
                    image=input_image,
                    time_length=video_duration,
                    num_steps=sample_steps,
                    guidance_weight=guidance_weight,
                    scheduler_scale=scheduler_scale,
                    expand_prompts=1,
                    save_path=output_filename,
                    seed=current_seed
                )
            else:
                global_pipe(
                    prompt,
                    time_length=video_duration,
                    width=width,
                    height=height,
                    num_steps=sample_steps,
                    guidance_weight=guidance_weight,
                    scheduler_scale=scheduler_scale,
                    expand_prompts=1,
                    save_path=output_filename,
                    seed=current_seed
                )

            elapsed = time.perf_counter() - start_time

            if os.path.exists(output_filename):
                all_generated_videos.append((output_filename, f"Seed: {current_seed}"))
                progress_msg = f"Completed {i+1}/{batch_size} in {elapsed:.1f}s"
                yield all_generated_videos.copy(), status_text, progress_msg
            else:
                yield all_generated_videos, f"Error: Output file not created for seed {current_seed}", ""

        except Exception as e:
            yield all_generated_videos, f"Error during generation: {str(e)}", ""
            return

    yield all_generated_videos, "All generations complete!", ""

def stop_generation():
    global stop_event
    stop_event.set()
    return "Stopping generation..."

def create_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Kandinsky 5.0 I2V Pro 20B - K1 Interface")

        with gr.Row():
            with gr.Column(scale=4):
                prompt = gr.Textbox(
                    scale=3,
                    label="Enter your prompt",
                    value="A cute tabby cat is eating a bowl of wasabi in a restaurant in Guangzhou. The cat is very good at using chopsticks and proceeds to eat the entire bowl of wasabi quickly with his chopsticks. The cat is wearing a white shirt with red accents and the cute tabby cat's shirt has the text 'spice kitten' on it. There is a large red sign in the background with '芥末' on it in white letters. A small red panda is drinking a beer beside the cat. The red panda is holding a large glass of dark beer and drinking it quickly. The panda tilts his head back and downs the entire glass of beer in one large gulp.",
                    lines=5
                )
                negative_prompt = gr.Textbox(
                    scale=3,
                    label="Negative Prompt",
                    value="Static, 2D cartoon, cartoon, 2d animation, paintings, images, worst quality, low quality, ugly, deformed, walking backwards",
                    lines=3,
                )
            with gr.Column(scale=1):
                batch_size = gr.Number(label="Batch Count", value=1, minimum=1, step=1)
            with gr.Column(scale=2):
                batch_progress = gr.Textbox(label="Status", interactive=False, value="")
                progress_text = gr.Textbox(label="Progress", interactive=False, value="")

        with gr.Row():
            generate_btn = gr.Button("Generate Video", elem_classes="green-btn")
            stop_btn = gr.Button("Stop Generation", variant="stop")

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Input Image (for i2v mode)", type="filepath")

                gr.Markdown("### Generation Parameters")
                mode = gr.Dropdown(
                    label="Mode",
                    choices=["i2v", "t2v"],
                    value="i2v",
                    info="Select generation mode: i2v (image-to-video) or t2v (text-to-video)"
                )
                with gr.Row():
                    width = gr.Number(label="Width", value=768, step=32, interactive=True)
                    height = gr.Number(label="Height", value=512, step=32, interactive=True)
                video_duration = gr.Slider(minimum=1, maximum=10, step=1, label="Video Duration (seconds)", value=5)
                sample_steps = gr.Slider(minimum=4, maximum=100, step=1, label="Sampling Steps", value=50)
                guidance_weight = gr.Slider(minimum=1.0, maximum=20.0, step=0.1, label="Guidance Weight", value=5.0)
                scheduler_scale = gr.Slider(minimum=0.0, maximum=20.0, step=0.1, label="Scheduler Scale", value=5.0)
                with gr.Row():
                    seed = gr.Number(label="Seed (-1 for random)", value=-1)
                    random_seed_btn = gr.Button("🎲")

            with gr.Column():
                output = gr.Gallery(
                    label="Generated Videos (Click to select)",
                    columns=[2], rows=[2], object_fit="contain", height="auto",
                    show_label=True, elem_id="gallery_k1", allow_preview=True, preview=True
                )

        with gr.Accordion("Model Settings & Performance", open=True):
            with gr.Row():
                enable_block_swap = gr.Checkbox(label="Enable Block Swap", value=True, info="Required for 24GB GPUs")
                blocks_in_memory = gr.Slider(minimum=1, maximum=60, step=1, label="Blocks in Memory", value=2, info="Number of transformer blocks to keep in GPU memory")
            with gr.Row():
                dtype_select = gr.Radio(choices=["bfloat16", "float16", "float32"], label="Data Type", value="bfloat16")
            with gr.Row():
                load_model_btn = gr.Button("Load Model", variant="primary")
                load_status = gr.Textbox(label="Model Status", interactive=False, value="Model not loaded")
            save_path = gr.Textbox(label="Save Path", value="outputs")

        random_seed_btn.click(
            fn=lambda: random.randint(0, 2**32 - 1),
            outputs=[seed]
        )

        load_model_btn.click(
            fn=load_pipeline,
            inputs=[mode, enable_block_swap, blocks_in_memory, dtype_select],
            outputs=[load_status]
        )

        generate_btn.click(
            fn=generate_video,
            inputs=[
                prompt, negative_prompt, input_image, mode,
                width, height, video_duration, sample_steps,
                guidance_weight, scheduler_scale, seed,
                enable_block_swap, blocks_in_memory, dtype_select,
                save_path, batch_size
            ],
            outputs=[output, batch_progress, progress_text]
        )

        stop_btn.click(
            fn=stop_generation,
            outputs=[batch_progress]
        )

    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
