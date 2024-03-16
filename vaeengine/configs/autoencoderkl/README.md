# AutoencoderKL

## Run Training

Run Training

```
# single gpu
$ vaeengine train ${CONFIG_FILE}
# multi gpus
$ NPROC_PER_NODE=${GPU_NUM} vaeengine train ${CONFIG_FILE}

# Example.
$ vaeengine train autoencoderkl_sdv15_pokemon
```

## Inference with diffusers

Once you have trained a model, specify the path to the saved model and utilize it for inference using the `diffusers.pipeline` module.

```py
from pathlib import Path

import torch
from diffusers import DiffusionPipeline, AutoencoderKL

checkpoint = Path('work_dirs/autoencoderkl_sdv15_pokemon/step627')
prompt = 'A yoda pokemon'

vae = AutoencoderKL.from_pretrained(
    checkpoint, subfolder="vae", torch_dtype=torch.float16)
pipe = DiffusionPipeline.from_pretrained(
    'runwayml/stable-diffusion-v1-5', torch_dtype=torch.float16)
pipe.to('cuda')

image = pipe(
    prompt,
    num_inference_steps=50
).images[0]
image.save('demo.png')
```
