## Example inference run

CUDA_VISIBLE_DEVICES=1 python test.py \
  --config ./configs/config_5s_i2v_pro_20b.yaml \
  --prompt "The Dragon breaths fire." \
  --image "./assets/test_image.jpg" \
  --video_duration 5 \
  --enable_block_swap \
  --seed 23982938 \
  --blocks_in_memory 6

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
