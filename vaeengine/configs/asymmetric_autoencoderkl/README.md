# Asymmetric AutoencoderKL

[Designing a Better Asymmetric VQGAN for StableDiffusion](https://arxiv.org/abs/2306.04632)

## Abstract

StableDiffusion is a revolutionary text-to-image generator that is causing a stir in the world of image generation and editing. Unlike traditional methods that learn a diffusion model in pixel space, StableDiffusion learns a diffusion model in the latent space via a VQGAN, ensuring both efficiency and quality. It not only supports image generation tasks, but also enables image editing for real images, such as image inpainting and local editing. However, we have observed that the vanilla VQGAN used in StableDiffusion leads to significant information loss, causing distortion artifacts even in non-edited image regions. To this end, we propose a new asymmetric VQGAN with two simple designs. Firstly, in addition to the input from the encoder, the decoder contains a conditional branch that incorporates information from task-specific priors, such as the unmasked image region in inpainting. Secondly, the decoder is much heavier than the encoder, allowing for more detailed recovery while only slightly increasing the total inference cost. The training cost of our asymmetric VQGAN is cheap, and we only need to retrain a new asymmetric decoder while keeping the vanilla VQGAN encoder and StableDiffusion unchanged. Our asymmetric VQGAN can be widely used in StableDiffusion-based inpainting and local editing methods. Extensive experiments demonstrate that it can significantly improve the inpainting and editing performance, while maintaining the original text-to-image capability.

<div align=center>
<img src="https://github.com/okotaku/vaeengine/assets/24734142/ec08b087-e641-40dc-ba55-847ed99a0f69"/>
</div>

## Citation

```
@misc{zhu2023designing,
      title={Designing a Better Asymmetric VQGAN for StableDiffusion},
      author={Zixin Zhu and Xuelu Feng and Dongdong Chen and Jianmin Bao and Le Wang and Yinpeng Chen and Lu Yuan and Gang Hua},
      year={2023},
      eprint={2306.04632},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Run Training

Run Training

```
# single gpu
$ vaeengine train ${CONFIG_FILE}
# multi gpus
$ NPROC_PER_NODE=${GPU_NUM} vaeengine train ${CONFIG_FILE}

# Example.
$ vaeengine train asymmetric_autoencoderkl_sdv15_pokemon
```

## Inference with diffusers

Once you have trained a model, specify the path to the saved model and utilize it for inference using the `diffusers.pipeline` module.

```py
from pathlib import Path

import torch
from diffusers import DiffusionPipeline, AutoencoderKL

checkpoint = Path('work_dirs/asymmetric_autoencoderkl_sdv15_pokemon/step627')
prompt = 'A yoda pokemon'

vae = AsymmetricAutoencoderKL.from_pretrained(
    checkpoint, subfolder="vae", torch_dtype=torch.float16)
pipe = DiffusionPipeline.from_pretrained(
    'runwayml/stable-diffusion-v1-5', vae=vae, torch_dtype=torch.float16)
pipe.to('cuda')

image = pipe(
    prompt,
    num_inference_steps=50
).images[0]
image.save('demo.png')
```
