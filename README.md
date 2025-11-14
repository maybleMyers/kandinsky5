## Example inference run
This will work on a 24gb gpu with like 80GB RAM used, could maybe lower duration to use less or something.  
To install I would use python 3.10 and torch 2.8.  
pip install torch==2.8.0+cu128 torchvision --index-url https://download.pytorch.org/whl/cu128  
you need to clone the full model into the kandinsky5 folder to make it work.  
git-lfs clone https://huggingface.co/kandinskylab/Kandinsky-5.0-I2V-Pro-sft-5s-Diffusers  

python test.py \  
  --config ./configs/config_5s_i2v_pro_20b.yaml \  
  --prompt "A cute tabby cat is eating a bowl of wasabi in a restaurant in Guangzhou. The cat is very good at using chopsticks and proceeds to   eat the entire bowl of wasabi quickly with his chopsticks. The cat is wearing a white shirt with red accents and the cute tabby cat's shirt has the text 'spice kitten' on it. There is a large red sign in the background with '芥末' on it in white letters. A small red panda is drinking a beer beside the cat. The red panda is holding a large glass of dark beer and drinking it quickly. The panda tilts his head back and downs the entire glass of beer in one large gulp." \
  --image "./assets/idekpero.png" \  
  --video_duration 5 \  
  --enable_block_swap \  
  --seed 67 \  
  --blocks_in_memory 2 \  
  --output_filename 5090wasabitcat.mp4 \  
  --sample_steps 50 \  
  --dtype bfloat16  

# Authors
<B>Project Leader:</B> Denis Dimitrov</br>

<B>Team Leads:</B> Vladimir Arkhipkin, Vladimir Korviakov, Nikolai Gerasimenko, Denis Parkhomenko</br>

<B>Core Contributors:</B> Alexey Letunovskiy, Maria Kovaleva, Ivan Kirillov, Lev Novitskiy, Denis Koposov, Dmitrii Mikhailov, Anna Averchenkova, Andrey Shutkin, Julia Agafonova, Olga Kim, Anastasiia Kargapoltseva, Nikita Kiselev</br>

<B>Contributors:</B> Anna Dmitrienko,  Anastasia Maltseva, Kirill Chernyshev, Ilia Vasiliev, Viacheslav Vasilev, Vladimir Polovnikov, Yury Kolabushin, Alexander Belykh, Mikhail Mamaev, Anastasia Aliaskina, Tatiana Nikulina, Polina Gavrilova</br>

# Citation

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

- [PyTorch](https://pytorch.org/) — for model training and inference.  
- [FlashAttention 3](https://github.com/Dao-AILab/flash-attention) — for efficient attention and faster inference.  
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen3-VL) — for providing high-quality text embeddings.  
- [CLIP](https://github.com/openai/CLIP) — for robust text–image alignment.  
- [HunyuanVideo](https://huggingface.co/tencent/HunyuanVideo) — for video latent encoding and decoding.  
- [MagCache](https://github.com/Zehong-Ma/MagCache) — for accelerated inference.
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) — for integration into node-based workflows.  

We deeply appreciate the contributions of these communities and researchers to the open-source ecosystem.
