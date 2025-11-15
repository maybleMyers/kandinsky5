import gradio as gr
from gradio import themes
from gradio.themes.utils import colors
import os
import sys
import time
import random
import subprocess
import re
from typing import Generator, List, Tuple, Optional, Dict
import threading
import json
import ffmpeg

stop_event = threading.Event()

def parse_progress_line(line: str) -> Optional[str]:
    """Parse progress bar lines and extract useful information."""
    line = line.strip()

    if "Loading checkpoint shards:" in line:
        match = re.search(r'(\d+)%.*?(\d+/\d+)', line)
        if match:
            percent = match.group(1)
            fraction = match.group(2)
            return f"Loading model: {percent}% ({fraction} shards)"

    if "Building DiT with block swapping" in line:
        return "Building DiT model..."

    if "Loading DiT weights from" in line:
        return "Loading DiT weights..."

    match = re.search(r'(\d+)%\|.*?\|\s*(\d+)/(\d+)\s*\[.*?<([\d:]+)', line)
    if match:
        percent = match.group(1)
        current = match.group(2)
        total = match.group(3)
        eta = match.group(4)
        return f"Generating: {percent}% ({current}/{total} steps) - ETA: {eta}"

    if "TIME ELAPSED:" in line:
        match = re.search(r'TIME ELAPSED:\s*([\d.]+)', line)
        if match:
            elapsed = float(match.group(1))
            return f"Generation completed in {elapsed:.1f}s"

    if "Generated video is saved to" in line:
        return "Video saved successfully!"

    return None

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
    global stop_event
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

        timestamp = int(time.time())
        output_filename = os.path.join(save_path, f"k1_{mode}_{timestamp}_{current_seed}.mp4")

        # Select config file based on mode
        if mode == "i2v":
            config_file = "./configs/config_5s_i2v_pro_20b.yaml"
        else:  # t2v
            config_file = "./configs/config_5s_t2v_pro_20b.yaml"

        command = [
            sys.executable, "test.py",
            "--config", config_file,
            "--prompt", str(prompt),
            "--video_duration", str(video_duration),
            "--sample_steps", str(sample_steps),
            "--seed", str(current_seed),
            "--output_filename", output_filename,
            "--dtype", dtype_str,
        ]

        if negative_prompt:
            command.extend(["--negative_prompt", str(negative_prompt)])

        if guidance_weight is not None:
            command.extend(["--guidance_weight", str(guidance_weight)])

        if scheduler_scale is not None:
            command.extend(["--scheduler_scale", str(scheduler_scale)])

        if mode == "i2v":
            if not input_image:
                yield all_generated_videos, "Error: Input image required for i2v mode.", ""
                return
            command.extend(["--image", str(input_image)])
        else:
            command.extend(["--width", str(int(width))])
            command.extend(["--height", str(int(height))])

        if enable_block_swap:
            command.append("--enable_block_swap")
            command.extend(["--blocks_in_memory", str(int(blocks_in_memory))])

        # Print the command for debugging/transparency
        print("\n" + "="*80)
        print(f"LAUNCHING COMMAND (Batch {i+1}/{batch_size}):")
        print(" ".join(command))
        print("="*80 + "\n")

        try:
            start_time = time.perf_counter()

            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            last_progress = ""
            while True:
                if stop_event.is_set():
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    yield all_generated_videos, "Generation stopped by user.", ""
                    return

                line = process.stdout.readline()
                if line:
                    print(line.strip())

                    parsed_progress = parse_progress_line(line)
                    if parsed_progress:
                        last_progress = parsed_progress
                        yield all_generated_videos.copy(), status_text, last_progress

                if process.poll() is not None:
                    break

            return_code = process.returncode

            elapsed = time.perf_counter() - start_time

            if return_code == 0 and os.path.exists(output_filename):
                # Save metadata to video
                params_for_meta = {
                    "model_type": "Kandinsky 5.0",
                    "mode": mode,
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "image_path": os.path.basename(input_image) if input_image else None,
                    "width": int(width) if mode == "t2v" else None,
                    "height": int(height) if mode == "t2v" else None,
                    "video_duration": video_duration,
                    "sample_steps": sample_steps,
                    "guidance_weight": guidance_weight,
                    "scheduler_scale": scheduler_scale,
                    "seed": current_seed,
                    "enable_block_swap": enable_block_swap,
                    "blocks_in_memory": int(blocks_in_memory) if enable_block_swap else None,
                    "dtype": dtype_str,
                    "config_file": config_file,
                }
                try:
                    add_metadata_to_video(output_filename, params_for_meta)
                    print(f"Added metadata to {output_filename}")
                except Exception as meta_err:
                    print(f"Warning: Failed to add metadata to {output_filename}: {meta_err}")

                all_generated_videos.append((output_filename, f"Seed: {current_seed}"))
                progress_msg = f"Completed {i+1}/{batch_size} in {elapsed:.1f}s"
                yield all_generated_videos.copy(), status_text, progress_msg
            else:
                error_msg = f"Error: Generation failed with return code {return_code}"
                yield all_generated_videos, error_msg, ""
                return

        except Exception as e:
            yield all_generated_videos, f"Error during generation: {str(e)}", ""
            return

    yield all_generated_videos, "All generations complete!", ""

def stop_generation():
    global stop_event
    stop_event.set()
    return "Stopping generation..."

def extract_video_metadata(video_path: str) -> Dict:
    """Extract metadata from video file using ffprobe."""
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        video_path
    ]

    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        metadata = json.loads(result.stdout.decode('utf-8'))
        if 'format' in metadata and 'tags' in metadata['format']:
            comment = metadata['format']['tags'].get('comment', '{}')
            return json.loads(comment)
        return {}
    except Exception as e:
        print(f"Metadata extraction failed: {str(e)}")
        return {}

def add_metadata_to_video(video_path: str, parameters: dict) -> None:
    """Add generation parameters to video metadata using ffmpeg."""
    # Convert parameters to JSON string
    params_json = json.dumps(parameters, indent=2)

    # Temporary output path
    temp_path = video_path.replace(".mp4", "_temp.mp4")

    # FFmpeg command to add metadata without re-encoding
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-metadata', f'comment={params_json}',
        '-codec', 'copy',
        temp_path
    ]

    try:
        # Execute FFmpeg command
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Replace original file with the metadata-enhanced version
        os.replace(temp_path, video_path)
    except subprocess.CalledProcessError as e:
        print(f"Failed to add metadata: {e.stderr.decode()}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
    except Exception as e:
        print(f"Error: {str(e)}")

def get_video_info(video_path: str) -> dict:
    """Get video information using ffmpeg-python."""
    try:
        probe = ffmpeg.probe(video_path)
        video_info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')

        width = int(video_info['width'])
        height = int(video_info['height'])
        fps = eval(video_info['r_frame_rate'])  # This converts '30/1' to 30.0

        # Calculate total frames
        duration = float(probe['format']['duration'])
        total_frames = int(duration * fps)

        return {
            'width': width,
            'height': height,
            'fps': fps,
            'total_frames': total_frames,
            'duration': duration
        }
    except Exception as e:
        print(f"Error extracting video info: {e}")
        return {}

def extract_video_details(video_path: str) -> Tuple[dict, str]:
    """Extract both metadata and video information."""
    metadata = extract_video_metadata(video_path)
    video_details = get_video_info(video_path)

    # Combine metadata with video details
    for key, value in video_details.items():
        if key not in metadata:
            metadata[key] = value

    # Return both the updated metadata and a status message
    return metadata, "Video details extracted successfully"

def create_interface():
    with gr.Blocks(
        theme=themes.Default(
            primary_hue=colors.Color(
                name="custom",
                c50="#E6F0FF",
                c100="#CCE0FF",
                c200="#99C1FF",
                c300="#66A3FF",
                c400="#3384FF",
                c500="#0060df",
                c600="#0052C2",
                c700="#003D91",
                c800="#002961",
                c900="#001430",
                c950="#000A18"
            )
        ),
        css="""
        .gallery-item:first-child { border: 2px solid #4CAF50 !important; }
        .gallery-item:first-child:hover { border-color: #45a049 !important; }
        .green-btn {
            background: linear-gradient(to bottom right, #2ecc71, #27ae60) !important;
            color: white !important;
            border: none !important;
        }
        .green-btn:hover {
            background: linear-gradient(to bottom right, #27ae60, #219651) !important;
        }
        .refresh-btn {
            max-width: 40px !important;
            min-width: 40px !important;
            height: 40px !important;
            border-radius: 50% !important;
            padding: 0 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        .light-blue-btn {
            background: linear-gradient(to bottom right, #AEC6CF, #9AB8C4) !important;
            color: #333 !important;
            border: 1px solid #9AB8C4 !important;
        }
        .light-blue-btn:hover {
            background: linear-gradient(to bottom right, #9AB8C4, #8AA9B5) !important;
            border-color: #8AA9B5 !important;
        }
        """,
    ) as demo:
        gr.Markdown("# Kandinsky 5.0 I2V Pro 20B - K1 Interface")

        with gr.Tab("Generation"):
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
                    sample_steps = gr.Slider(minimum=1, maximum=100, step=1, label="Sampling Steps", value=50)
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
                save_path = gr.Textbox(label="Save Path", value="outputs")

            random_seed_btn.click(
                fn=lambda: random.randint(0, 2**32 - 1),
                outputs=[seed]
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

        # Video Info Tab
        with gr.Tab("Video Info"):
            with gr.Row():
                video_input = gr.Video(label="Upload Video", interactive=True)
                metadata_output = gr.JSON(label="Generation Parameters")

            with gr.Row():
                video_info_status = gr.Textbox(label="Status", interactive=False)

            video_input.upload(
                fn=extract_video_details,
                inputs=video_input,
                outputs=[metadata_output, video_info_status]
            )

    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", share=False)
