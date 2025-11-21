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
import io
from PIL import Image
import tiktoken

# Initialize tiktoken encoder for fast token counting
enc = tiktoken.get_encoding("cl100k_base")

def count_tokens(text):
    """Count tokens in text using tiktoken."""
    if not text:
        return 0
    return len(enc.encode(text))

stop_event = threading.Event()
current_process = None  # Track the currently running process
current_output_filename = None  # Track current output filename for early stop signals

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
    model_config: str,
    dit_checkpoint_path: str,
    attention_engine: str,
    attention_type: str,
    nabla_P: float,
    nabla_wT: int,
    nabla_wW: int,
    nabla_wH: int,
    width: int,
    height: int,
    video_duration: int,
    sample_steps: int,
    guidance_weight: float,
    scheduler_scale: float,
    seed: int,
    use_mixed_weights: bool,
    use_int8: bool,
    use_torch_compile: bool,
    use_magcache: bool,
    enable_block_swap: bool,
    blocks_in_memory: int,
    dtype_str: str,
    text_encoder_dtype_str: str,
    vae_dtype_str: str,
    computation_dtype_str: str,
    save_path: str,
    batch_size: int,
    enable_preview: bool,
    preview_steps: int,
    enable_vae_chunking: bool,
    vae_temporal_tile_frames: int,
    vae_temporal_stride_frames: int,
    vae_spatial_tile_height: int,
    vae_spatial_tile_width: int,
) -> Generator[Tuple[List[Tuple[str, str]], Optional[str], str, str], None, None]:
    global stop_event, current_process, current_output_filename
    stop_event.clear()
    current_process = None
    current_output_filename = None

    os.makedirs(save_path, exist_ok=True)
    all_generated_videos = []

    for i in range(int(batch_size)):
        if stop_event.is_set():
            current_process = None
            yield all_generated_videos, None, "Generation stopped by user.", ""
            return

        current_seed = seed
        if seed == -1:
            current_seed = random.randint(0, 2**32 - 1)
        elif int(batch_size) > 1:
            current_seed = seed + i

        status_text = f"Processing {i+1}/{batch_size} (Seed: {current_seed})"
        yield all_generated_videos.copy(), None, status_text, "Starting generation..."

        timestamp = int(time.time())
        run_id = f"{timestamp}_{random.randint(1000, 9999)}"
        unique_preview_suffix = f"k1_{run_id}"
        output_filename = os.path.join(save_path, f"k1_{mode}_{timestamp}_{current_seed}.mp4")
        current_output_filename = output_filename  # Track for early stop signals

        # Select config file based on model_config selection
        config_map = {
            "5s Lite (T2V)": "./configs/config_5s_sft.yaml",
            "10s Lite (T2V)": "./configs/config_10s_sft.yaml",
            "5s Pro 20B (T2V)": "./configs/config_5s_t2v_pro_20b.yaml",
            "5s Pro 20B HD (T2V)": "./configs/k5_pro_t2v_5s_sft_hd.yaml",
            "10s Pro 20B (T2V)": "./configs/config_10s_t2v_pro_20b.yaml",
            "10s Pro 20B HD (T2V)": "./configs/k5_pro_t2v_10s_sft_hd.yaml",
            "5s Pro 20B (I2V)": "./configs/config_5s_i2v_pro_20b.yaml",
            "5s Pro 20B HD (I2V)": "./configs/k5_pro_i2v_5s_sft_hd.yaml",
            "5s Lite (I2V)": "./configs/config_5s_i2v.yaml",
        }

        config_file = config_map.get(model_config, "./configs/config_5s_t2v_pro_20b.yaml")

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

        # Add attention engine if specified
        if attention_engine and attention_engine != "auto":
            command.extend(["--attention_engine", attention_engine])

        # Add DiT checkpoint path if specified
        if dit_checkpoint_path and dit_checkpoint_path.strip():
            command.extend(["--checkpoint_path", dit_checkpoint_path.strip()])

        # Add attention type and NABLA parameters if specified
        if attention_type and attention_type != "auto":
            command.extend(["--attention_type", attention_type])
            if attention_type == "nabla":
                # NABLA requires 128-pixel alignment for i2v
                if mode == "i2v":
                    width = (int(width) // 128) * 128
                    height = (int(height) // 128) * 128
                    width = max(128, width)
                    height = max(128, height)

                command.extend(["--nabla_P", str(nabla_P)])
                command.extend(["--nabla_wT", str(int(nabla_wT))])
                command.extend(["--nabla_wW", str(int(nabla_wW))])
                command.extend(["--nabla_wH", str(int(nabla_wH))])
                command.append("--nabla_add_sta")

        if text_encoder_dtype_str:
            command.extend(["--text_encoder_dtype", text_encoder_dtype_str])
        if vae_dtype_str:
            command.extend(["--vae_dtype", vae_dtype_str])
        if computation_dtype_str:
            command.extend(["--computation_dtype", computation_dtype_str])

        if use_mixed_weights:
            command.append("--use_mixed_weights")
        if use_int8:
            command.append("--use_int8")
        if not use_torch_compile:
            command.append("--no_compile")
        if use_magcache:
            command.append("--magcache")

        if negative_prompt:
            command.extend(["--negative_prompt", str(negative_prompt)])

        if guidance_weight is not None:
            command.extend(["--guidance_weight", str(guidance_weight)])

        if scheduler_scale is not None:
            command.extend(["--scheduler_scale", str(scheduler_scale)])

        if mode == "i2v":
            if not input_image:
                current_process = None
                yield all_generated_videos, "Error: Input image required for i2v mode.", ""
                return
            command.extend(["--image", str(input_image)])
            # Pass width and height for i2v mode to resize the input image
            command.extend(["--width", str(int(width))])
            command.extend(["--height", str(int(height))])
        else:
            command.extend(["--width", str(int(width))])
            command.extend(["--height", str(int(height))])

        if enable_block_swap:
            command.append("--enable_block_swap")
            command.extend(["--blocks_in_memory", str(int(blocks_in_memory))])

        if enable_preview and preview_steps > 0:
            command.extend(["--preview", str(preview_steps)])
            command.extend(["--preview_suffix", unique_preview_suffix])

        # Add VAE chunking parameters if enabled
        if enable_vae_chunking:
            if vae_temporal_tile_frames and vae_temporal_tile_frames > 0:
                command.extend(["--vae_temporal_tile_frames", str(int(vae_temporal_tile_frames))])
            if vae_temporal_stride_frames and vae_temporal_stride_frames > 0:
                command.extend(["--vae_temporal_stride_frames", str(int(vae_temporal_stride_frames))])
            if vae_spatial_tile_height and vae_spatial_tile_height > 0:
                command.extend(["--vae_spatial_tile_height", str(int(vae_spatial_tile_height))])
            if vae_spatial_tile_width and vae_spatial_tile_width > 0:
                command.extend(["--vae_spatial_tile_width", str(int(vae_spatial_tile_width))])

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

            current_process = process

            current_preview_yield_path = None
            last_preview_mtime = 0
            preview_base_dir = os.path.join(save_path, "previews")
            preview_mp4_path = os.path.join(preview_base_dir, f"latent_preview_{unique_preview_suffix}.mp4")

            last_progress = ""
            while True:
                if stop_event.is_set():
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    current_process = None
                    yield all_generated_videos, None, "Generation stopped by user.", ""
                    return

                line = process.stdout.readline()
                if line:
                    print(line.strip())

                    parsed_progress = parse_progress_line(line)
                    if parsed_progress:
                        last_progress = parsed_progress

                if enable_preview:
                    if os.path.exists(preview_mp4_path):
                        current_mtime = os.path.getmtime(preview_mp4_path)
                        if current_mtime > last_preview_mtime:
                            current_preview_yield_path = preview_mp4_path
                            last_preview_mtime = current_mtime

                yield all_generated_videos.copy(), current_preview_yield_path, status_text, last_progress

                if process.poll() is not None:
                    break

            # Clear current process when done
            current_process = None
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
                    "width": int(width),
                    "height": int(height),
                    "video_duration": video_duration,
                    "sample_steps": sample_steps,
                    "guidance_weight": guidance_weight,
                    "scheduler_scale": scheduler_scale,
                    "seed": current_seed,
                    "use_mixed_weights": use_mixed_weights,
                    "use_int8": use_int8,
                    "use_torch_compile": use_torch_compile,
                    "use_magcache": use_magcache,
                    "enable_block_swap": enable_block_swap,
                    "blocks_in_memory": int(blocks_in_memory) if enable_block_swap else None,
                    "dtype": dtype_str,
                    "text_encoder_dtype": text_encoder_dtype_str if text_encoder_dtype_str else None,
                    "vae_dtype": vae_dtype_str if vae_dtype_str else None,
                    "computation_dtype": computation_dtype_str if computation_dtype_str else None,
                    "config_file": config_file,
                    "model_config": model_config,
                    "dit_checkpoint_path": dit_checkpoint_path if dit_checkpoint_path and dit_checkpoint_path.strip() else None,
                    "attention_engine": attention_engine,
                    "attention_type": attention_type if attention_type != "auto" else None,
                    "nabla_P": nabla_P if attention_type == "nabla" else None,
                    "nabla_wT": int(nabla_wT) if attention_type == "nabla" else None,
                    "nabla_wW": int(nabla_wW) if attention_type == "nabla" else None,
                    "nabla_wH": int(nabla_wH) if attention_type == "nabla" else None,
                    "enable_vae_chunking": enable_vae_chunking,
                    "vae_temporal_tile_frames": int(vae_temporal_tile_frames) if enable_vae_chunking and vae_temporal_tile_frames else None,
                    "vae_temporal_stride_frames": int(vae_temporal_stride_frames) if enable_vae_chunking and vae_temporal_stride_frames else None,
                    "vae_spatial_tile_height": int(vae_spatial_tile_height) if enable_vae_chunking and vae_spatial_tile_height else None,
                    "vae_spatial_tile_width": int(vae_spatial_tile_width) if enable_vae_chunking and vae_spatial_tile_width else None,
                }
                try:
                    add_metadata_to_video(output_filename, params_for_meta)
                    print(f"Added metadata to {output_filename}")
                except Exception as meta_err:
                    print(f"Warning: Failed to add metadata to {output_filename}: {meta_err}")

                all_generated_videos.append((output_filename, f"Seed: {current_seed}"))
                progress_msg = f"Completed {i+1}/{batch_size} in {elapsed:.1f}s"
                yield all_generated_videos.copy(), None, status_text, progress_msg
            else:
                error_msg = f"Error: Generation failed with return code {return_code}"
                current_process = None
                yield all_generated_videos, None, error_msg, ""
                return

        except Exception as e:
            current_process = None
            yield all_generated_videos, None, f"Error during generation: {str(e)}", ""
            return

    current_process = None
    yield all_generated_videos, None, "All generations complete!", ""

def stop_generation():
    global stop_event, current_process
    stop_event.set()

    # Immediately terminate the current process if it exists
    if current_process is not None:
        try:
            current_process.terminate()
            # Give it a moment to terminate gracefully
            try:
                current_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't terminate
                current_process.kill()
                current_process.wait()
        except Exception as e:
            print(f"Error stopping process: {e}")

    return "Stopping generation..."

def stop_and_decode():
    """Signal the generation to stop and decode the current latents."""
    global current_output_filename
    if current_output_filename:
        signal_file = current_output_filename + ".stop_decode"
        try:
            with open(signal_file, 'w') as f:
                f.write('decode')
            return "Signaling stop & decode..."
        except Exception as e:
            return f"Error creating signal file: {e}"
    return "No active generation to stop"

def stop_and_save():
    """Signal the generation to stop and save the current latents."""
    global current_output_filename
    if current_output_filename:
        signal_file = current_output_filename + ".stop_save"
        try:
            with open(signal_file, 'w') as f:
                f.write('save')
            return "Signaling stop & save latents..."
        except Exception as e:
            return f"Error creating signal file: {e}"
    return "No active generation to stop"


def resume_from_checkpoint(
    checkpoint_path: str,
    model_config: str,
    save_path: str,
    use_torch_compile: bool,
    enable_block_swap: bool,
    blocks_in_memory: int,
    dtype_str: str,
    vae_dtype_str: str,
) -> Generator[Tuple[List[Tuple[str, str]], Optional[str], str, str], None, None]:
    """Resume generation from a saved checkpoint."""
    global stop_event, current_process, current_output_filename
    stop_event.clear()
    current_process = None

    if not checkpoint_path or not os.path.exists(checkpoint_path):
        yield [], None, "Error: No checkpoint file selected", ""
        return

    os.makedirs(save_path, exist_ok=True)

    # Generate output filename
    timestamp = int(time.time())
    output_filename = os.path.join(save_path, f"k1_resume_{timestamp}.mp4")
    current_output_filename = output_filename

    # Select config file based on model_config
    config_map = {
        "5s Lite (T2V)": "./configs/config_5s_sft.yaml",
        "5s Pro 20B (T2V)": "./configs/k5_pro_t2v_5s_sft.yaml",
        "5s Pro 20B HD (T2V)": "./configs/k5_pro_t2v_5s_sft_hd.yaml",
        "5s Lite (I2V)": "./configs/config_5s_sft_i2v.yaml",
        "5s Pro 20B (I2V)": "./configs/k5_pro_i2v_5s_sft.yaml",
        "5s Pro 20B HD (I2V)": "./configs/k5_pro_i2v_5s_sft_hd.yaml",
    }
    config_file = config_map.get(model_config, "./configs/k5_pro_i2v_5s_sft.yaml")

    yield [], None, "Resuming from checkpoint...", f"Loading {checkpoint_path}"

    # Build command
    command = [
        "python", "test.py",
        "--config", config_file,
        "--prompt", "resume",  # Placeholder, not used in resume mode
        "--output_filename", output_filename,
        "--resume_from", checkpoint_path,
    ]

    # Add dtype settings
    if dtype_str:
        command.extend(["--dtype", dtype_str])
    if vae_dtype_str:
        command.extend(["--vae_dtype", vae_dtype_str])

    # Add block swap settings
    if enable_block_swap:
        command.append("--enable_block_swap")
        command.extend(["--blocks_in_memory", str(int(blocks_in_memory))])

    # Add compile setting
    if not use_torch_compile:
        command.append("--no_compile")

    # Print command for debugging
    print(f">>> Resume command: {' '.join(command)}", flush=True)

    try:
        current_process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        for line in iter(current_process.stdout.readline, ''):
            if stop_event.is_set():
                break

            line = line.strip()
            if not line:
                continue

            # Print all output to console for debugging
            print(line, flush=True)

            progress_info = parse_progress_line(line)
            if progress_info:
                yield [], None, "Resuming...", progress_info
            elif ">>>" in line or "Error" in line or "error" in line:
                # Show important messages
                yield [], None, "Resuming...", line

        current_process.wait()

        if os.path.exists(output_filename):
            yield [(output_filename, f"Resumed video")], None, "Resume complete!", ""
        else:
            # Check for new checkpoint
            new_checkpoint = output_filename.replace(".mp4", "_checkpoint.pt")
            if os.path.exists(new_checkpoint):
                yield [], None, f"Checkpoint saved: {new_checkpoint}", ""
            else:
                yield [], None, "Resume stopped", ""

    except Exception as e:
        yield [], None, f"Error: {str(e)}", ""
    finally:
        current_process = None


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
    """Get video information using ffprobe via subprocess (no python-ffmpeg dependency)."""
    try:
        # Select first video stream and get specific entries
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,r_frame_rate,duration',
            '-of', 'json',
            video_path
        ]
        
        # Run ffprobe
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        data = json.loads(result.stdout)
        
        if not data.get('streams'):
            return {}
            
        video_stream = data['streams'][0]
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        
        # Calculate FPS (handle fraction like 30/1)
        r_frame_rate = video_stream.get('r_frame_rate', '30/1')
        if '/' in r_frame_rate:
            num, den = map(int, r_frame_rate.split('/'))
            fps = num / den if den != 0 else 0
        else:
            fps = float(r_frame_rate)

        # Calculate Duration
        # Try stream duration first, then format duration
        duration = float(video_stream.get('duration', 0))
        if duration == 0:
            # Fallback to format duration if stream duration is missing
            cmd_fmt = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', video_path]
            res_fmt = subprocess.run(cmd_fmt, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            data_fmt = json.loads(res_fmt.stdout)
            duration = float(data_fmt.get('format', {}).get('duration', 0))

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

def send_to_generation(video_path, metadata):
    """Extract first frame and map metadata to generation inputs using subprocess."""
    if not video_path:
        return [gr.update()] * 35

    # 1. Extract First Frame as PIL Image using subprocess
    input_img_pil = None
    try:
        # ffmpeg command to output 1st frame to stdout as PNG
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vframes', '1',
            '-f', 'image2',
            '-c:v', 'png',
            'pipe:1'
        ]
        
        # Run command and capture binary output
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        
        if result.stdout:
            input_img_pil = Image.open(io.BytesIO(result.stdout))
            
    except Exception as e:
        print(f"Frame extraction failed: {e}")

    # 2. Parse Metadata
    meta = metadata if isinstance(metadata, dict) else {}

    def get(k, default=gr.update()):
        return meta.get(k, default)
    
    def get_num(k, default=gr.update()):
        val = meta.get(k)
        return val if val is not None else default

    # 3. Return values to populate fields and switch tab
    return (
        gr.Tabs(selected="gen_tab"),    # Switch to Generation tab
        get("prompt"),                  # prompt
        get("negative_prompt"),         # negative_prompt
        input_img_pil,                  # input_image (First Frame)
        "i2v",                          # mode (Force to i2v)
        get("model_config"),            # model_config
        get("attention_engine"),        # attention_engine
        get("attention_type"),          # attention_type
        get("dit_checkpoint_path"),     # dit_checkpoint_path
        get_num("nabla_P"),             # nabla_P
        get_num("nabla_wT"),            # nabla_wT
        get_num("nabla_wW"),            # nabla_wW
        get_num("nabla_wH"),            # nabla_wH
        get_num("width"),               # width
        get_num("height"),              # height
        get_num("video_duration"),      # video_duration
        get_num("sample_steps"),        # sample_steps
        get_num("guidance_weight"),     # guidance_weight
        get_num("scheduler_scale"),     # scheduler_scale
        get_num("seed"),                # seed
        get("use_mixed_weights"),       # use_mixed_weights
        get("use_int8"),                # use_int8
        get("use_torch_compile"),       # use_torch_compile
        get("use_magcache"),            # use_magcache
        get("enable_block_swap"),       # enable_block_swap
        get_num("blocks_in_memory"),    # blocks_in_memory
        get("dtype"),                   # dtype_select
        get("text_encoder_dtype"),      # text_encoder_dtype_select
        get("vae_dtype"),               # vae_dtype_select
        get("computation_dtype"),       # computation_dtype_select
        get("enable_vae_chunking"),     # enable_vae_chunking
        get_num("vae_temporal_tile_frames"),   # vae_temporal_tile_frames
        get_num("vae_temporal_stride_frames"), # vae_temporal_stride_frames
        get_num("vae_spatial_tile_height"),    # vae_spatial_tile_height
        get_num("vae_spatial_tile_width")      # vae_spatial_tile_width
    )

def calculate_width_from_height(height, original_dims):
    """Calculate width based on height maintaining aspect ratio (divisible by 64)"""
    if not original_dims or height is None:
        return gr.update()
    try:
        # Ensure height is an integer and divisible by 64
        height = int(height)
        if height <= 0:
            return gr.update()
        height = (height // 64) * 64
        height = max(64, height)  # Min height (64 is divisible by 64)

        orig_w, orig_h = map(int, original_dims.split('x'))
        if orig_h == 0:
            return gr.update()
        aspect_ratio = orig_w / orig_h
        # Calculate new width, rounding to the nearest multiple of 64
        new_width = round((height * aspect_ratio) / 64) * 64
        return gr.update(value=max(64, new_width))  # Ensure minimum size

    except Exception as e:
        print(f"Error calculating width: {e}")
        return gr.update()

def calculate_height_from_width(width, original_dims):
    """Calculate height based on width maintaining aspect ratio (divisible by 64)"""
    if not original_dims or width is None:
        return gr.update()
    try:
        # Ensure width is an integer and divisible by 64
        width = int(width)
        if width <= 0:
            return gr.update()
        width = (width // 64) * 64
        width = max(64, width)  # Min width (64 is divisible by 64)

        orig_w, orig_h = map(int, original_dims.split('x'))
        if orig_w == 0:
            return gr.update()
        aspect_ratio = orig_w / orig_h
        # Calculate new height, rounding to the nearest multiple of 64
        new_height = round((width / aspect_ratio) / 64) * 64
        return gr.update(value=max(64, new_height))  # Ensure minimum size

    except Exception as e:
        print(f"Error calculating height: {e}")
        return gr.update()

def update_resolution_from_scale(scale, original_dims):
    """Update dimensions based on scale percentage (divisible by 64)"""
    if not original_dims:
        return gr.update(), gr.update()
    try:
        scale = float(scale) if scale is not None else 100.0
        if scale <= 0:
            scale = 100.0

        orig_w, orig_h = map(int, original_dims.split('x'))
        scale_factor = scale / 100.0

        # Calculate and round to the nearest multiple of 64
        new_w = round((orig_w * scale_factor) / 64) * 64
        new_h = round((orig_h * scale_factor) / 64) * 64

        # Ensure minimum size (must be multiple of 64)
        new_w = max(64, new_w)  # 64 is divisible by 64
        new_h = max(64, new_h)

        return gr.update(value=new_w), gr.update(value=new_h)
    except Exception as e:
        print(f"Error updating from scale: {e}")
        return gr.update(), gr.update()

def update_image_dimensions(image_path):
    """Update original dimensions when image is uploaded"""
    if image_path is None:
        return "", gr.update(), gr.update()
    try:
        img = Image.open(image_path)
        w, h = img.size
        original_dims_str = f"{w}x{h}"
        # Calculate dimensions snapped to nearest multiple of 64 while maintaining aspect ratio
        new_w = round(w / 64) * 64
        new_h = round(h / 64) * 64
        new_w = max(64, new_w)
        new_h = max(64, new_h)
        return original_dims_str, gr.update(value=new_w), gr.update(value=new_h)
    except Exception as e:
        print(f"Error reading image dimensions: {e}")
        return "", gr.update(), gr.update()

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
        demo.load(None, None, None, js=r"""
            () => {
                document.title = 'Kandinsky 5 - K1';

                function updateTitle(text) {
                    if (text && text.trim()) {
                        // Match k1.py's format: "Generating: XX% (current/total steps) - ETA: HH:MM:SS"
                        // Also support raw TQDM format: "XX%|...[...<HH:MM:SS"
                        // Also support h1111 format: "(XX%)" + "ETA: HH:MM:SS"
                        const pattern = /(?:.*?(\d+)%.*?(?:ETA|Remaining):\s*([\d:]+))|(?:(\d+)%\|.*\[.*<([\d:?]+))/;
                        const match = text.match(pattern);

                        if (match) {
                            const percentage = match[1] || match[3];
                            const time = match[2] || match[4];
                            if (percentage && time) {
                                 document.title = `[${percentage}% ETA: ${time}] - K1`;
                            }
                        }
                    }
                }

                setTimeout(() => {
                    const progressElements = document.querySelectorAll('textarea.scroll-hide');
                    progressElements.forEach(element => {
                        if (element) {
                            new MutationObserver(() => {
                                updateTitle(element.value);
                            }).observe(element, {
                                attributes: true,
                                childList: true,
                                characterData: true,
                                subtree: true
                            });
                        }
                    });
                }, 1000);
            }
            """)

        with gr.Tabs() as tabs:
            with gr.Tab("Generation", id="gen_tab"):
                with gr.Row():
                    with gr.Column(scale=4):
                        prompt = gr.Textbox(
                            scale=3,
                            label="Enter your prompt",
                            value="A cute tabby cat is eating a bowl of wasabi in a restaurant in Guangzhou. The cat is very good at using chopsticks and proceeds to eat the entire bowl of wasabi quickly with his chopsticks. The cat is wearing a white shirt with red accents and the cute tabby cat's shirt has the text 'spice kitten' on it. There is a large red sign in the background with '芥末' on it in white letters. A small red panda is drinking a beer beside the cat. The red panda is holding a large glass of dark beer and drinking it quickly. The panda tilts his head back and downs the entire glass of beer in one large gulp.",
                            lines=5
                        )
                        token_count_display = gr.Textbox(
                            label="Token Count",
                            value=str(count_tokens("A cute tabby cat is eating a bowl of wasabi in a restaurant in Guangzhou. The cat is very good at using chopsticks and proceeds to eat the entire bowl of wasabi quickly with his chopsticks. The cat is wearing a white shirt with red accents and the cute tabby cat's shirt has the text 'spice kitten' on it. There is a large red sign in the background with '芥末' on it in white letters. A small red panda is drinking a beer beside the cat. The red panda is holding a large glass of dark beer and drinking it quickly. The panda tilts his head back and downs the entire glass of beer in one large gulp.")),
                            interactive=False,
                            scale=1
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

                        model_config = gr.Dropdown(
                            label="Model Configuration",
                            choices=[
                                "5s Lite (T2V)",
                                "10s Lite (T2V)",
                                "5s Pro 20B (T2V)",
                                "5s Pro 20B HD (T2V)",
                                "10s Pro 20B (T2V)",
                                "10s Pro 20B HD (T2V)",
                                "5s Pro 20B (I2V)",
                                "5s Pro 20B HD (I2V)",
                                "5s Lite (I2V)"
                            ],
                            value="5s Pro 20B (I2V)",
                            info="Select model configuration. Pro models require more VRAM but offer better quality. HD uses NABLA sparse attention for 1024+ resolution."
                        )

                        attention_engine = gr.Dropdown(
                            label="Attention Engine",
                            choices=["auto", "flash_attention_2", "flash_attention_3", "sdpa", "sage"],
                            value="auto",
                            info="Select attention implementation. 'auto' uses config default. Flash attention is faster for Lite models. SDPA works well for Pro models."
                        )

                        with gr.Accordion("NABLA Sparse Attention Settings", open=False):
                            gr.Markdown("""
                            Configure NABLA sparse attention for memory-efficient long video generation. Recommended for 10s models.

                            **Important for i2v:** NABLA requires resolution divisible by 128 pixels (e.g., 512, 640, 768, 1024, 1280, 1536, 1920, 2048).
                            Invalid resolutions will be automatically rounded down to the nearest multiple of 128.
                            """)
                            attention_type = gr.Dropdown(
                                label="Attention Type",
                                choices=["auto", "flash", "nabla"],
                                value="auto",
                                info="'auto' uses config default, 'flash' for full attention, 'nabla' for sparse attention (better for 10s videos)"
                            )
                            with gr.Row():
                                nabla_P = gr.Slider(
                                    minimum=0.5, maximum=1.0, value=0.9, step=0.05,
                                    label="NABLA P (Probability Threshold)",
                                    info="Top-k probability threshold. Higher = more tokens kept (0.9 recommended)"
                                )
                            with gr.Row():
                                nabla_wT = gr.Slider(
                                    minimum=3, maximum=21, value=11, step=2,
                                    label="NABLA wT (Temporal Window)",
                                    info="Temporal window size. Use 11 for 10s, 7 for 5s"
                                )
                                nabla_wW = gr.Slider(
                                    minimum=1, maximum=7, value=3, step=2,
                                    label="NABLA wW (Width Window)",
                                    info="Width window size (default: 3)"
                                )
                                nabla_wH = gr.Slider(
                                    minimum=1, maximum=7, value=3, step=2,
                                    label="NABLA wH (Height Window)",
                                    info="Height window size (default: 3)"
                                )

                        dit_checkpoint_path = gr.Textbox(
                            label="DiT Checkpoint Path (optional)",
                            value="",
                            placeholder="./weights/model/kandinsky5pro_t2v_sft_10s.safetensors",
                            info="Override DiT model checkpoint path. Leave empty to use config default. Provide path to your .safetensors file."
                        )

                        # Hidden state to store original image dimensions
                        original_dims = gr.State(value="")

                        gr.Markdown("### Resolution Settings")
                        scale_slider = gr.Slider(
                            minimum=1, maximum=200, value=100, step=1,
                            label="Scale % (adjusts resolution while maintaining aspect ratio)",
                            info="Scale the input image dimensions. Works for both i2v and t2v modes."
                        )
                        with gr.Row():
                            width = gr.Number(label="Width", value=768, step=64, interactive=True,
                                            info="Must be divisible by 64")
                            calc_height_btn = gr.Button("→", size="sm")
                            calc_width_btn = gr.Button("←", size="sm")
                            height = gr.Number(label="Height", value=512, step=64, interactive=True,
                                            info="Must be divisible by 64")

                        video_duration = gr.Slider(minimum=1, maximum=30, step=1, label="Video Duration (seconds)", value=5)
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
                        with gr.Accordion("Latent Preview (During Generation)", open=True):
                            enable_preview = gr.Checkbox(label="Enable Latent Preview", value=True)
                            preview_steps = gr.Slider(minimum=1, maximum=50, step=1, value=5,
                                                    label="Preview Every N Steps")
                            preview_output = gr.Video(
                                label="Latest Preview", height=300,
                                interactive=False, elem_id="k1_preview_video"
                            )
                        stop_decode_btn = gr.Button("Stop & Decode", elem_classes="light-blue-btn")
                        stop_save_btn = gr.Button("Stop & Save Latents", elem_classes="light-blue-btn")
                        checkpoint_file = gr.Textbox(
                            label="Checkpoint File (for resume)",
                            placeholder="Path to _checkpoint.pt file",
                            scale=3
                        )
                        resume_btn = gr.Button("Resume from Checkpoint", elem_classes="light-blue-btn", scale=1)                            

                with gr.Accordion("Model Settings & Performance", open=True):
                    with gr.Row():
                        use_mixed_weights = gr.Checkbox(label="Use Mixed Weights", value=False, info="Preserve fp32 for critical layers (norms, embeddings)")
                        use_int8 = gr.Checkbox(label="Use int8 matmul", value=False, info="enable int8 quantization")
                        use_torch_compile = gr.Checkbox(label="Use torch.compile", value=False, info="Slower startup (2-5 min) but faster inference")
                        use_magcache = gr.Checkbox(label="Use MagCache", value=False, info="Skip redundant computations (50-step models only)")
                    with gr.Row():
                        enable_block_swap = gr.Checkbox(label="Enable Block Swap", value=True, info="Required for 24GB GPUs")
                        blocks_in_memory = gr.Slider(minimum=1, maximum=60, step=1, label="Blocks in Memory", value=2, info="Number of transformer blocks to keep in GPU memory")
                    with gr.Row():
                        dtype_select = gr.Radio(choices=["bfloat16", "float16", "float32", "fp8_scaled"], label="Default Data Type", value="bfloat16", info="Used for all components if specific dtypes not set. fp8_scaled provides ~50% memory savings.")
                    with gr.Accordion("Advanced: Component-Specific Data Types", open=False):
                        gr.Markdown("Override dtypes for individual components. Leave empty to use default dtype.")
                        with gr.Row():
                            text_encoder_dtype_select = gr.Dropdown(choices=["", "bfloat16", "float16", "float32", "fp8_scaled"], label="Text Encoder dtype", value="", info="Empty = use default")
                            vae_dtype_select = gr.Dropdown(choices=["", "bfloat16", "float16", "float32", "fp8_scaled"], label="VAE dtype", value="", info="Empty = use default")
                            computation_dtype_select = gr.Dropdown(choices=["", "bfloat16", "float16", "float32", "fp8_scaled"], label="Computation dtype", value="", info="Empty = use default. fp8_scaled for transformer only.")

                    with gr.Accordion("Advanced: VAE Memory Optimization (Chunking)", open=False):
                        gr.Markdown("""
                        **Configure VAE temporal/spatial chunking to reduce memory usage during decode.**

                        Enable this if you get OOM (Out of Memory) errors during VAE decode. Smaller chunk sizes use less memory but take longer to decode.

                        - **Temporal Tile Frames**: Chunk size in frames (default: 16). Try 12 for moderate reduction, 8 for aggressive.
                        - **Temporal Stride**: Overlap between chunks (default: auto = tile_frames - 4)
                        - **Spatial Tile Height/Width**: Spatial chunk dimensions (default: 256)

                        Leave disabled to use default settings (recommended unless you experience OOM).
                        """)
                        enable_vae_chunking = gr.Checkbox(
                            label="Enable VAE Chunking",
                            value=False,
                            info="Enable custom VAE chunk sizes to reduce memory usage"
                        )
                        with gr.Row():
                            vae_temporal_tile_frames = gr.Slider(
                                minimum=4, maximum=32, value=12, step=4,
                                label="Temporal Tile Frames",
                                info="Chunk size in pixel-space frames (must be divisible by 4)"
                            )
                            vae_temporal_stride_frames = gr.Slider(
                                minimum=0, maximum=28, value=0, step=4,
                                label="Temporal Stride Frames (0 = auto)",
                                info="Overlap between chunks. 0 = auto-calculate as tile_frames - 4"
                            )
                        with gr.Row():
                            vae_spatial_tile_height = gr.Slider(
                                minimum=128, maximum=512, value=256, step=64,
                                label="Spatial Tile Height",
                                info="Spatial chunk height (reduce if high resolution causes OOM)"
                            )
                            vae_spatial_tile_width = gr.Slider(
                                minimum=128, maximum=512, value=256, step=64,
                                label="Spatial Tile Width",
                                info="Spatial chunk width (reduce if high resolution causes OOM)"
                            )

                    save_path = gr.Textbox(label="Save Path", value="outputs")

                random_seed_btn.click(
                    fn=lambda: (-1),
                    outputs=[seed]
                )

                # Token count update - real-time as user types
                prompt.change(
                    fn=lambda text: str(count_tokens(text)),
                    inputs=[prompt],
                    outputs=[token_count_display]
                )

                # Resolution control event handlers
                input_image.change(
                    fn=update_image_dimensions,
                    inputs=[input_image],
                    outputs=[original_dims, width, height]
                )

                scale_slider.change(
                    fn=update_resolution_from_scale,
                    inputs=[scale_slider, original_dims],
                    outputs=[width, height]
                )

                calc_width_btn.click(
                    fn=calculate_width_from_height,
                    inputs=[height, original_dims],
                    outputs=[width]
                )

                calc_height_btn.click(
                    fn=calculate_height_from_width,
                    inputs=[width, original_dims],
                    outputs=[height]
                )

                generate_btn.click(
                    fn=generate_video,
                    inputs=[
                        prompt, negative_prompt, input_image, mode, model_config, dit_checkpoint_path, attention_engine,
                        attention_type, nabla_P, nabla_wT, nabla_wW, nabla_wH,
                        width, height, video_duration, sample_steps,
                        guidance_weight, scheduler_scale, seed,
                        use_mixed_weights, use_int8, use_torch_compile, use_magcache, enable_block_swap, blocks_in_memory, dtype_select,
                        text_encoder_dtype_select, vae_dtype_select, computation_dtype_select,
                        save_path, batch_size,
                        enable_preview, preview_steps,
                        enable_vae_chunking, vae_temporal_tile_frames, vae_temporal_stride_frames,
                        vae_spatial_tile_height, vae_spatial_tile_width
                    ],
                    outputs=[output, preview_output, batch_progress, progress_text]
                )

                stop_btn.click(
                    fn=stop_generation,
                    outputs=[batch_progress]
                )

                stop_decode_btn.click(
                    fn=stop_and_decode,
                    outputs=[batch_progress]
                )

                stop_save_btn.click(
                    fn=stop_and_save,
                    outputs=[batch_progress]
                )

                resume_btn.click(
                    fn=resume_from_checkpoint,
                    inputs=[
                        checkpoint_file, model_config, save_path,
                        use_torch_compile, enable_block_swap, blocks_in_memory,
                        dtype_select, vae_dtype_select
                    ],
                    outputs=[output, preview_output, batch_progress, progress_text]
                )

            # Video Info Tab
            with gr.Tab("Video Info"):
                with gr.Row():
                    video_input = gr.Video(label="Upload Video", interactive=True)
                    metadata_output = gr.JSON(label="Generation Parameters")

                with gr.Row():
                    video_info_status = gr.Textbox(label="Status", interactive=False)

                send_to_gen_btn = gr.Button("Send to Generation 🚀", elem_classes="green-btn")

                video_input.upload(
                    fn=extract_video_details,
                    inputs=video_input,
                    outputs=[metadata_output, video_info_status]
                )

                send_to_gen_btn.click(
                    fn=send_to_generation,
                    inputs=[video_input, metadata_output],
                    outputs=[
                        tabs,                       # 1. Switch Tab
                        prompt,                     # 2
                        negative_prompt,            # 3
                        input_image,                # 4
                        mode,                       # 5
                        model_config,               # 6
                        attention_engine,           # 7
                        attention_type,             # 8
                        dit_checkpoint_path,        # 9
                        nabla_P,                    # 10
                        nabla_wT,                   # 11
                        nabla_wW,                   # 12
                        nabla_wH,                   # 13
                        width,                      # 14
                        height,                     # 15
                        video_duration,             # 16
                        sample_steps,               # 17
                        guidance_weight,            # 18
                        scheduler_scale,            # 19
                        seed,                       # 20
                        use_mixed_weights,          # 21
                        use_int8,                   # 22
                        use_torch_compile,          # 23
                        use_magcache,               # 24
                        enable_block_swap,          # 25
                        blocks_in_memory,           # 26
                        dtype_select,               # 27
                        text_encoder_dtype_select,  # 28
                        vae_dtype_select,           # 29
                        computation_dtype_select,   # 30
                        enable_vae_chunking,        # 31
                        vae_temporal_tile_frames,   # 32
                        vae_temporal_stride_frames, # 33
                        vae_spatial_tile_height,    # 34
                        vae_spatial_tile_width      # 35
                    ]
                )

    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", share=False)