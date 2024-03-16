from diffusers import AutoencoderKL

from vaeengine.models.editors import AutoencoderKLModel

model = dict(
   type=AutoencoderKLModel,
   vae=dict(
      type=AutoencoderKL.from_pretrained,
      pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
      subfolder="vae"))
