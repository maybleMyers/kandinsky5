## Example inference run
This will work on a 24gb gpu with like 80GB RAM used, could maybe lower duration to use less or something.  

To install I would use python 3.10 and torch 2.8. Linux is much less painless for you to use.    
python3.10 -m venv env  
pip install torch==2.8.0+cu128 torchvision --index-url https://download.pytorch.org/whl/cu128  
pip install -r requirements.txt   
you need to clone the full model into the kandinsky5 folder to make it work.  
git-lfs clone https://huggingface.co/kandinskylab/Kandinsky-5.0-I2V-Pro-sft-5s-Diffusers  

to use the gui:  
source env/bin/activate  (or env/scripts/activate if you are a windows powershell console)  
python k1.py  
open browser goto 127.0.0.1:7860

I added some mixed models here: https://huggingface.co/maybleMyers/kan/  
You can input them in the DiT Checkpoint Path in the gui.  

Windows install is hard, I would recommend activating linux before trying to run.  

## To use lite i2v 
download https://huggingface.co/kandinskylab/Kandinsky-5.0-I2V-Lite-5s/blob/main/model/kandinsky5lite_i2v_5s.safetensors that model and put it in the lite_checkpoints subfolder. You need to have the full i2v pro diffusers cloned in the root directory. Select mode - i2v and model configuration 5s Lite (I2V) .  Either mess with the vae config or set 1 block swapped for now to offload before vae decoding.  


Changlog:  
  11/18/2025  
    Add support for lite i2v https://huggingface.co/kandinskylab/Kandinsky-5.0-I2V-Lite-5s/tree/main/model  
    make previews better    
  11/17/2025  
    Add preview support. Add int8 support to drastically lower ram/vram reqs.  


### Authors
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
    howpublished = {\url{https://github.com/ai-forever/Kandinsky-5}},
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
