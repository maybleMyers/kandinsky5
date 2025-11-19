<div align="center">
  <a href="https://habr.com/ru/companies/sberbank/articles/951800/">Habr</a> | <a href="https://kandinskylab.ai/">Project Page</a> | Technical Report (soon) | 🤗 <a href=https://huggingface.co/collections/kandinskylab/kandinsky-50-video-lite> Video Lite </a> / <a href=https://huggingface.co/collections/kandinskylab/kandinsky-50-video-pro> Video Pro </a> / <a href=https://huggingface.co/collections/kandinskylab/kandinsky-50-image-lite> Image Lite </a> | <a href="https://huggingface.co/docs/diffusers/main/en/api/pipelines/kandinsky5"> 🤗 Diffusers </a>  | <a href="https://github.com/kandinskylab/kandinsky-5/blob/main/comfyui/README.md">ComfyUI</a>
</div>

---

## Custom Fork - Quick Start Guide

This fork includes enhanced features for VAE tiling, memory management, and GUI improvements.

### Quick Installation (24GB GPU, ~80GB RAM recommended)

To install I would use python 3.10 and torch 2.8. Linux is much less painless for you to use.
```bash
python3.10 -m venv env
pip install torch==2.8.0+cu128 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

You need to clone the full model into the kandinsky5 folder to make it work:
```bash
git-lfs clone https://huggingface.co/kandinskylab/Kandinsky-5.0-I2V-Pro-sft-5s-Diffusers
```

### Using the GUI

```bash
source env/bin/activate  # or env/scripts/activate on Windows PowerShell
python k1.py
```
Then open browser and goto http://127.0.0.1:7860

I added some mixed models here: https://huggingface.co/maybleMyers/kan/
You can input them in the DiT Checkpoint Path in the gui.

**Note:** Windows install is hard, I would recommend using Linux for better experience.

### To use lite i2v

Download the model from https://huggingface.co/kandinskylab/Kandinsky-5.0-I2V-Lite-5s/blob/main/model/kandinsky5lite_i2v_5s.safetensors and put it in the `lite_checkpoints` subfolder. You need to have the full i2v pro diffusers cloned in the root directory. Select mode - i2v and model configuration 5s Lite (I2V). Either mess with the vae config or set 1 block swapped for now to offload before vae decoding.

### Changelog (Custom Fork)

- **11/18/2025**
  - Add support for lite i2v https://huggingface.co/kandinskylab/Kandinsky-5.0-I2V-Lite-5s/tree/main/model
  - Make previews better
- **11/17/2025**
  - Add preview support
  - Add int8 support to drastically lower RAM/VRAM requirements

---

## Official Kandinsky 5.0 Documentation

- 🔥 ```2025/11/15```: `Kandinsky 5.0 Lite I2V` & `Kandinsky 5.0 Lite T2I` models are open-sourced.
- 🔥 ```2025/10/19```: Further VAE tiling optimization. NF4 version of Qwen2.5-VL from Bitsandbytes is supported. Flash Attention 2, Flash Attention 2, Sage Attention or SDPA can be selected for 5-seconds generation using option --attention_engine. Now generation should work on the GPUS with 12 GB of memory. Kandinsky 5 Video Lite is [accepted to diffusers](https://github.com/huggingface/diffusers/pull/12478).
- 🔥 ```2025/10/7```: The ComfyUI README file has been updated. SDPA support has been added, allowing you to run our code without Flash attention. Magcache support for nocfg checkpoints has been added, allowing Magcache support for sft and nocfg checkpoints. Memory consumption in the VAE has been reduced, with the entire pipeline now running at 24 GB with offloading.
- 🔥 ```2025/09/29```: We have open-sourced `Kandinsky 5.0 T2V Lite` a lite (2B parameters) version of `Kandinsky 5.0 Video` text-to-video generation model. Released checkpoints: `kandinsky5lite_t2v_pretrain_5s`, `kandinsky5lite_t2v_pretrain_10s`, `kandinsky5lite_t2v_sft_5s`, `kandinsky5lite_t2v_sft_10s`, `kandinsky5lite_t2v_nocfg_5s`, `kandinsky5lite_t2v_nocfg_10s`, `kandinsky5lite_t2v_distilled16steps_5s`, `kandinsky5lite_t2v_distilled16steps_10s` contains weight from pretrain, supervised finetuning, cfg distillation and diffusion distillation into 16 steps. 5s checkpoints are capable of generating videos up to 5 seconds long. 10s checkpoints is faster models checkpoints trained with [NABLA](https://huggingface.co/ai-forever/Wan2.1-T2V-14B-NABLA-0.7) algorithm and capable to generate videos up to 10 seconds long.

## Kandinsky 5.0 Video Lite

Kandinsky 5.0 Video Lite is a lightweight video generation model (2B parameters) that ranks #1 among open-source models in its class. It outperforms larger Wan models (5B and 14B) and offers the best understanding of Russian concepts in the open-source ecosystem.

We provide 8 Text-to-Video model variants, each optimized for different use cases:

* SFT model — delivers the highest generation quality;

* CFG-distilled — runs 2× faster;

* Diffusion-distilled — enables low-latency generation with minimal quality loss (6× faster);

* Pretrain model — designed for fine-tuning by researchers and enthusiasts.

All models are available in two versions: for generating 5-second and 10-second videos.

Additionally, we provide Image-to-Video model capable to generate video given input image and textt prompt.

## Kandinsky 5.0 Image Lite

Kandinsky 5.0 Image Lite is a 6B image generation model with the following capabilities:

* 1K resulution (1280x768, 1024x1024 and others).

* High visual quality

* Strong text-writing

* Russian concepts understanding

## Pipeline

**Latent diffusion pipeline** with **Flow Matching**.

**Diffusion Transformer (DiT)** as the main generative backbone with **cross-attention to text embeddings**.

- **Qwen2.5-VL** and **CLIP** provides text embeddings.

- **HunyuanVideo 3D VAE** encodes/decodes video into a latent space.

- **DiT** is the main generative module using cross-attention to condition on text.

<img width="1600" height="477" alt="Picture1" src="https://github.com/user-attachments/assets/17fc2eb5-05e3-4591-9ec6-0f6e1ca397b3" />

<img width="800" height="406" alt="Picture2" src="https://github.com/user-attachments/assets/f3006742-e261-4c39-b7dc-e39330be9a09" />


## Model Zoo

| Model                               | config | video duration | NFE | Checkpoint | Latency* |
|-------------------------------------|--------|----------------|-----|------------|----------------|
| Kandinsky 5.0 T2V Lite SFT 5s       |configs/config_5s_sft.yaml | 5s             | 100 |🤗 [HF](https://huggingface.co/kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s) |      139 s     |
| Kandinsky 5.0 T2V Lite SFT 10s      |configs/config_10s_sft.yaml| 10s            | 100 |🤗 [HF](https://huggingface.co/kandinskylab/Kandinsky-5.0-T2V-Lite-sft-10s) |      224 s     |
| Kandinsky 5.0 T2V Lite pretrain 5s  |configs/config_5s_pretrain.yaml | 5s             | 100 |🤗 [HF](https://huggingface.co/kandinskylab/Kandinsky-5.0-T2V-Lite-pretrain-5s) |      139 s      |
| Kandinsky 5.0 T2V Lite pretrain 10s |configs/config_10s_pretrain.yaml | 10s            | 100 |🤗 [HF](https://huggingface.co/kandinskylab/Kandinsky-5.0-T2V-Lite-pretrain-10s) |     224 s      |
| Kandinsky 5.0 T2V Lite no-CFG 5s    |configs/config_5s_nocfg.yaml| 5s             | 50  |🤗 [HF](https://huggingface.co/kandinskylab/Kandinsky-5.0-T2V-Lite-nocfg-5s) |       77 s     |
| Kandinsky 5.0 T2V Lite no-CFG 10s   |configs/config_10s_nocfg.yaml| 10s            | 50  |🤗 [HF](https://huggingface.co/kandinskylab/Kandinsky-5.0-T2V-Lite-nocfg-10s) |     124 s      |
| Kandinsky 5.0 T2V Lite distill 5s   |configs/config_5s_distil.yaml| 5s             | 16  | 🤗 [HF](https://huggingface.co/kandinskylab/Kandinsky-5.0-T2V-Lite-distilled16steps-5s)|       35 s     |
| Kandinsky 5.0 T2V Lite distill 10s  |configs/config_10s_distil.yaml| 10s            | 16  | 🤗 [HF](https://huggingface.co/kandinskylab/Kandinsky-5.0-T2V-Lite-distilled16steps-10s)|      61 s      |
| Kandinsky 5.0 I2V Lite 5s  |configs/config_5s_i2v.yaml| 5s            | 100  | 🤗 [HF](https://huggingface.co/kandinskylab/Kandinsky-5.0-I2V-Lite-5s)|      139 s      |
| Kandinsky 5.0 T2I Lite  |configs/config_t2i.yaml| -           | 100  | 🤗 [HF](https://huggingface.co/kandinskylab/Kandinsky-5.0-T2I-Lite)|      13 s      |

*Latency was measured after the second inference run. The first run of the model can be slower due to the compilation process. Inference was measured on an NVIDIA H100 GPU with 80 GB of memory, using CUDA 12.8.1 and PyTorch 2.8. For 5-second models Flash Attention 3 was used.

### Examples:

#### Kandinsky 5.0 T2V Lite SFT

<table border="0" style="width: 200; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/bc38821b-f9f1-46db-885f-1f70464669eb" width=200 controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/9f64c940-4df8-4c51-bd81-a05de8e70fc3" width=200 controls autoplay loop></video>
      </td>
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/77dd417f-e0bf-42bd-8d80-daffcd054add" width=200 controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/385a0076-f01c-4663-aa46-6ce50352b9ed" width=200 controls autoplay loop></video>
      </td>
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/7c1bcb31-cc7d-4385-9a33-2b0cc28393dd" width=200 controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/990a8a0b-2df1-4bbc-b2e3-2859b6f1eea6" width=200 controls autoplay loop></video>
      </td>
  </tr>

</table>


#### Kandinsky 5.0 T2V Lite Distill

<table border="0" style="width: 200; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/861342f9-f576-4083-8a3b-94570a970d58" width=200 controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/302e4e7d-781d-4a58-9b10-8c473d469c4b" width=200 controls autoplay loop></video>
      </td>
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/3e70175c-40e5-4aec-b506-38006fe91a76" width=200 controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/b7da85f7-8b62-4d46-9460-7f0e505de810" width=200 controls autoplay loop></video>
      </td>

</table>


#### Kandinsky 5.0 T2I Lite

<table border="0" style="width: 200; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <image src="https://github.com/user-attachments/assets/f46e6866-15ce-445d-bb81-9843a341e2a9" width=200 ></image>
      </td>
      <td>
          <image src="https://github.com/user-attachments/assets/74f3af1f-b11e-4174-9f36-e956b871a6e6" width=200 ></image>
      </td>
      <td>
          <image src="https://github.com/user-attachments/assets/7e469d09-8b96-4691-b929-dd809827adf9" width=200 ></image>
      </td>
  <tr>
</table>
<table border="0" style="width: 200; text-align: left; margin-top: 10px;">
      <td>
          <image src="https://github.com/user-attachments/assets/8054b25b-5d71-4547-8822-b07d71d137f4" width=200 ></image>
      </td>
      <td>
          <image src="https://github.com/user-attachments/assets/f4825237-640b-4b2d-86e6-fd08fe95039f" width=200 ></image>
      </td>
      <td>
          <image src="https://github.com/user-attachments/assets/73fbbc2a-3249-4b70-8931-2893ab0107a5" width=200 ></image>
      </td>

</table>
<table border="0" style="width: 200; text-align: left; margin-top: 10px;">
      <td>
          <image src="https://github.com/user-attachments/assets/c309650b-8d8b-4e44-bb63-48287e22ff44" width=200 ></image>
      </td>
      <td>
          <image src="https://github.com/user-attachments/assets/d5c0fcca-69b7-4d77-9c36-cd2fb87f2615" width=200 ></image>
      </td>
      <td>
          <image src="https://github.com/user-attachments/assets/7895c3e8-2e72-40b8-8bf7-dcac859a6b29" width=200 ></image>
      </td>

</table>

### Results:

#### Side-by-Side evaluation

The evaluation is based on the expanded prompts from the [Movie Gen benchmark](https://github.com/facebookresearch/MovieGenBench), which are available in the expanded_prompt column of the benchmark/moviegen_bench.csv file.

<table border="0" style="width: 400; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <img src="assets/sbs/kandinsky_5_video_lite_vs_sora.jpg" width=400 ></img>
      </td>
      <td>
          <img src="assets/sbs/kandinsky_5_video_lite_vs_wan_2.1_14B.jpg" width=400 ></img>
      </td>
  <tr>
      <td>
          <img src="assets/sbs/kandinsky_5_video_lite_vs_wan_2.2_5B.jpg" width=400 ></img>
      </td>
      <td>
          <img src="assets/sbs/kandinsky_5_video_lite_vs_wan_2.2_A14B.jpg" width=400 ></img>
      </td>
  <tr>
      <td>
          <img src="assets/sbs/kandinsky_5_video_lite_vs_wan_2.1_1.3B.jpg" width=400 ></img>
      </td>

</table>

#### Distill Side-by-Side evaluation

<table border="0" style="width: 400; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <img src="assets/sbs/kandinsky_5_video_lite_5s_vs_kandinsky_5_video_lite_distill_5s.jpg" width=400 ></img>
      </td>
      <td>
          <img src="assets/sbs/kandinsky_5_video_lite_10s_vs_kandinsky_5_video_lite_distill_10s.jpg" width=400 ></img>
      </td>

</table>

#### VBench results

<div align="center">
  <picture>
    <img src="assets/vbench.png">
  </picture>
</div>

## Quickstart

#### Installation
Clone the repo:
```sh
git clone https://github.com/kandinskylab/kandinsky-5.git
cd kandinsky-5
```

Install dependencies:
```sh
pip install -r requirements.txt
```

To improve inference performance on NVidia Hopper GPUs, we recommend installing [Flash Attention 3](https://github.com/Dao-AILab/flash-attention/?tab=readme-ov-file#flashattention-3-beta-release).

#### Model Download
```sh
python download_models.py
```

#### Run Kandinsky 5.0 T2V Lite SFT 5s

```sh
python test.py --prompt "A dog in red hat"
```

#### Run Kandinsky 5.0 T2V Lite SFT 10s 

```sh
python test.py --config ./configs/config_10s_sft.yaml --prompt "A dog in red hat" --video_duration 10 
```

#### Run Kandinsky 5.0 T2V Lite pretrain 5s

```sh
python test.py --config ./configs/config_5s_pretrain.yaml --prompt "A dog in red hat"
```

#### Run Kandinsky 5.0 T2V Lite pretrain 10s

```sh
python test.py --config ./configs/config_10s_pretrain.yaml --prompt "A dog in red hat" --video_duration 10
```

#### Run Kandinsky 5.0 T2V Lite no-CFG 5s

```sh
python test.py --config ./configs/config_5s_nocfg.yaml --prompt "A dog in red hat" 
```

#### Run Kandinsky 5.0 T2V Lite no-CFG 10s

```sh
python test.py --config ./configs/config_10s_nocfg.yaml --prompt "A dog in red hat" --video_duration 10
```

#### Run Kandinsky 5.0 T2V Lite distill 5s

```sh
python test.py --config ./configs/config_5s_distil.yaml --prompt "A dog in red hat"          
```

#### Run Kandinsky 5.0 T2V Lite distill 10s

```sh
python test.py --config ./configs/config_10s_distil.yaml --prompt "A dog in red hat" --video_duration 10
```

#### Run Kandinsky 5.0 I2V Lite 5s

```sh
python test.py --config configs/config_5s_i2v.yaml --prompt "The Dragon breaths fire." --image "./assets/test_image.jpg" --video_duration 5
```

#### Run Kandinsky 5.0 T2I Lite

```sh
python test.py --config ./configs/config_t2i.yaml --prompt "A dog in a red hat" --width=1280 --height=768
```

### T2V Inference

```python
import torch
from kandinsky import get_T2V_pipeline

device_map = {
    "dit": torch.device('cuda:0'), 
    "vae": torch.device('cuda:0'), 
    "text_embedder": torch.device('cuda:0')
}

pipe = get_T2V_pipeline(device_map, conf_path="configs/config_5s_sft.yaml")

images = pipe(
    seed=42,
    time_length=5,
    width=768,
    height=512,
    save_path="./test.mp4",
    text="A cat in a red hat",
)
```

### I2V Inference

```python
import torch
from kandinsky import get_I2V_pipeline

device_map = {
    "dit": torch.device('cuda:0'), 
    "vae": torch.device('cuda:0'), 
    "text_embedder": torch.device('cuda:0')
}

pipe = get_I2V_pipeline(device_map, conf_path="configs/config_5s_i2v.yaml")

images = pipe(
    seed=42,
    time_length=5,
    save_path='./test.mp4',
    text="The Dragon breaths fire.",
    image = "assets/test_image.jpg",
)
```

### T2I Inference

```python
import torch
from kandinsky import get_T2I_pipeline

device_map = {
    "dit": torch.device('cuda:0'), 
    "vae": torch.device('cuda:0'), 
    "text_embedder": torch.device('cuda:0')
}

pipe = get_T2I_pipeline(device_map, conf_path="./configs/config_t2i.yaml")

images = pipe(
    seed=42,
    save_path='./test.png',
    text="A cat in a red hat with a label 'HELLO'"
)
```

Please, refer to [inference_example.ipynb](inference_example.ipynb)/[inference_example_i2v.ipynb](inference_example_i2v.ipynb)/[inference_example_i2v.ipynb](inference_example_t2i.ipynb) notebooks for more usage details.

### Distributed Inference

For a faster inference, we also provide the capability to perform inference in a distributed way:
```
NUMBER_OF_NODES=1
NUMBER_OF_DEVICES_PER_NODE=1 / 2 / 4
python -m torch.distributed.launch --nnodes $NUMBER_OF_NODES --nproc-per-node $NUMBER_OF_DEVICES_PER_NODE test.py
```

### Optimized Inference

#### Offloading
For less memory consumption you can use **offloading** of the models.
```sh
python test.py --prompt "A dog in red hat" --offload
```

#### Magcache
Also we provide [Magcache](https://github.com/Zehong-Ma/MagCache) inference for faster generations (now available for sft 5s and sft 10s checkpoints).

```sh
python test.py --prompt "A dog in red hat" --magcache
```

#### Qwen encoder quantization
To reduce GPU memory needed for Qwen encoder we provide option to use NF4-quantized version from [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes).

```sh
python test.py --prompt "A dog in red hat" --qwen_quantization
```

#### Attention engine selection
Depending on your hardware you can use the follwing full attention algorithm implementation:
* PyTorch [SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
* [Flash Attention 2](https://github.com/Dao-AILab/flash-attention)
* [Flash Attention 3](https://github.com/Dao-AILab/flash-attention/tree/main/hopper)
* [Sage Attention](https://github.com/thu-ml/SageAttention)

The attention algorithm can be selected using an option "--attention_engine" of test.py script for 5 second (and less) video generation. For 10-second generation we use sparse attention algorithm [NABLA](https://arxiv.org/abs/2507.13546).

Note that currently (19 Oct. 2025) version build from source contains a bug and produces noisy output. A temporary workaround to fix it is decribed [here](https://github.com/thu-ml/SageAttention/issues/277).

```sh
python test.py --prompt "A dog in red hat" --attention_engine=flash_attention_3
```

```sh
python test.py --prompt "A dog in red hat" --attention_engine=flash_attention_2
```

```sh
python test.py --prompt "A dog in red hat" --attention_engine=sdpa
```

```sh
python test.py --prompt "A dog in red hat" --attention_engine=sage
```

By default we use option --attention_engine=auto which enables automatic selection of the most optimal algorithm installed in your system.

### ComfyUI

See the instruction [here](comfyui)

### CacheDiT

cache-dit offers Fully Cache Acceleration support for Kandinsky-5 with DBCache, TaylorSeer and Cache CFG. Visit their [example](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline/run_kandinsky5_t2v.py) for more details.

### Beta testing
You can apply to participate in the beta testing of the Kandinsky Video Lite via the [telegram bot](https://t.me/kandinsky_access_bot).

## 📑 Todo List
- Kandinsky 5.0 Lite Text-to-Video
    - [x] Multi-GPU Inference code of the 2B models
    - [ ] Checkpoints 2B models
      - [x] pretrain
      - [x] sft
      - [ ] rl
      - [x] cfg distil 
      - [x] distil 16 steps
      - [ ] autoregressive generation
    - [x] ComfyUI integration
    - [x] Diffusers integration
    - [x] Caching acceleration support
- Kandinsky 5.0 Lite Image-to-Video
    - [x] Multi-GPU Inference code of the 2B model
    - [x] Checkpoints of the 2B model
    - [x] ComfyUI integration
    - [ ] Diffusers integration
- Kandinsky 5.0 Pro Text-to-Video
    - [ ] Multi-GPU Inference code of the models
    - [ ] Checkpoints of the model
    - [ ] ComfyUI integration
    - [ ] Diffusers integration
- Kandinsky 5.0 Pro Image-to-Video
    - [ ] Multi-GPU Inference code of the model
    - [ ] Checkpoints of the model
    - [ ] ComfyUI integration
    - [ ] Diffusers integration
- Kandinsky 5.0 Lite Text-to-Image
    - [ ] Checkpoints of the model
      - [x] Image generation
      - [ ] Image editing
    - [ ] ComfyUI integration
    - [ ] Diffusers integration
    - [x] Caching acceleration support
- [ ] Technical report

# Authors

<B>Project Leader:</B> Denis Dimitrov</br>

<B>Team Leads:</B> Vladimir Arkhipkin, Vladimir Korviakov, Nikolai Gerasimenko, Denis Parkhomenko</br>

<B>Core Contributors:</B> Alexey Letunovskiy, Maria Kovaleva, Ivan Kirillov, Lev Novitskiy, Denis Koposov, Dmitrii Mikhailov, Anna Averchenkova, Andrey Shutkin, Julia Agafonova, Olga Kim, Anastasiia Kargapoltseva, Nikita Kiselev</br>

<B>Contributors:</B> Anna Dmitrienko,  Anastasia Maltseva, Kirill Chernyshev, Ilia Vasiliev, Viacheslav Vasilev, Vladimir Polovnikov, Yury Kolabushin, Alexander Belykh, Mikhail Mamaev, Anastasia Aliaskina, Tatiana Nikulina, Polina Gavrilova</br>

### Citation

```
@misc{kandinsky2025,
    author = {Alexey Letunovskiy, Maria Kovaleva, Ivan Kirillov, Lev Novitskiy, Denis Koposov,
              Dmitrii Mikhailov, Anna Averchenkova, Andrey Shutkin, Julia Agafonova, Olga Kim,
              Anastasiia Kargapoltseva, Nikita Kiselev, Vladimir Arkhipkin, Vladimir Korviakov,
              Nikolai Gerasimenko, Denis Parkhomenko, Anna Dmitrienko, Anastasia Maltseva,
              Kirill Chernyshev, Ilia Vasiliev, Viacheslav Vasilev, Vladimir Polovnikov,
              Yury Kolabushin, Alexander Belykh, Mikhail Mamaev, Anastasia Aliaskina,
              Tatiana Nikulina, Polina Gavrilova, Denis Dimitrov},
    title = {Kandinsky 5.0: A family of diffusion models for Video & Image generation},
    howpublished = {\url{https://github.com/kandinskylab/kandinsky-5}},
    year = 2025
}

@misc{mikhailov2025nablanablaneighborhoodadaptiveblocklevel,
      title={$\nabla$NABLA: Neighborhood Adaptive Block-Level Attention}, 
      author={Dmitrii Mikhailov and Aleksey Letunovskiy and Maria Kovaleva and Vladimir Arkhipkin
              and Vladimir Korviakov and Vladimir Polovnikov and Viacheslav Vasilev
              and Evelina Sidorova and Denis Dimitrov},
      year={2025},
      eprint={2507.13546},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.13546}, 
}
```

# Acknowledgements

We gratefully acknowledge the open-source projects and research that made Kandinsky 5.0 possible:

- [INT8 Suport](https://github.com/lodestone-rock/RamTorch) — for int8 drop in support and triton kernel.
- [PyTorch](https://pytorch.org/) — for model training and inference.  
- [FlashAttention 3](https://github.com/Dao-AILab/flash-attention) — for efficient attention and faster inference.  
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen3-VL) — for providing high-quality text embeddings.  
- [CLIP](https://github.com/openai/CLIP) — for robust text–image alignment.  
- [HunyuanVideo](https://huggingface.co/tencent/HunyuanVideo) — for video latent encoding and decoding.  
- [MagCache](https://github.com/Zehong-Ma/MagCache) — for accelerated inference.
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) — for integration into node-based workflows.  

We deeply appreciate the contributions of these communities and researchers to the open-source ecosystem.
