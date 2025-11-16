import os
import argparse
from huggingface_hub import snapshot_download

MODELS = ['ai-forever/Kandinsky-5.0-T2V-Lite-pretrain-5s',
          'ai-forever/Kandinsky-5.0-T2V-Lite-pretrain-10s',
          'ai-forever/Kandinsky-5.0-T2V-Lite-sft-5s',
          'ai-forever/Kandinsky-5.0-T2V-Lite-sft-10s',
          'ai-forever/Kandinsky-5.0-T2V-Lite-nocfg-5s',
          'ai-forever/Kandinsky-5.0-T2V-Lite-nocfg-10s',
          'ai-forever/Kandinsky-5.0-T2V-Lite-distilled16steps-5s',
          'ai-forever/Kandinsky-5.0-T2V-Lite-distilled16steps-10s',
          'ai-forever/Kandinsky-5.0-I2V-Lite-5s',
          'kandinskylab/Kandinsky-5.0-T2I-Lite',
          ]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default='all')
    args = parser.parse_args()

    models = MODELS if args.models == 'all' else args.models.split(',')
    for model in models:
        assert model in MODELS, f'unknown model {model}'
    cache_dir = "./weights"
    
    for model in models:
        dit_path = snapshot_download(
            repo_id=f"ai-forever/{model}",
            allow_patterns="model/*",
            local_dir=cache_dir,
        )
    if any(['-I2V-' in model or '-T2V-' in model]):
        vae_path = snapshot_download(
            repo_id="hunyuanvideo-community/HunyuanVideo",
            allow_patterns="vae/*",
            local_dir=cache_dir,
        )
    
    if any(['-I2I-' in model or '-T2I-' in model]):
        vae_path = snapshot_download(
            repo_id="black-forest-labs/FLUX.1-dev",
            allow_patterns="vae/*",
            local_dir=cache_dir,
        )

    text_encoder_path = snapshot_download(
        repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
        local_dir=os.path.join(cache_dir, "text_encoder/"),
    )

    text_encoder2_path = snapshot_download(
        repo_id="openai/clip-vit-large-patch14",
        local_dir=os.path.join(cache_dir, "text_encoder2/"),
    )

    
