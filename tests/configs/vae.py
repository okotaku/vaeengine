from diffusers import AutoencoderKL

from vaeengine.models.editors import AutoencoderKLModel

base_model = "diffusers/tiny-stable-diffusion-torch"
model = dict(
            type=AutoencoderKLModel,
            vae=dict(
               type=AutoencoderKL.from_pretrained,
               pretrained_model_name_or_path=base_model,
               subfolder="vae"))
