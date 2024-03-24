from diffusers import AutoencoderKL

from vaeengine.models.editors import AutoencoderKLModel
from vaeengine.models.losses import LPIPSLoss

model = dict(
   type=AutoencoderKLModel,
   vae=dict(
      type=AutoencoderKL.from_pretrained,
      pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
      subfolder="vae"),
   lpips_loss=dict(type=LPIPSLoss, loss_weight=0.1))
