from diffusers import AsymmetricAutoencoderKL

from vaeengine.models.editors import AsymmetricAutoencoderKLModel
from vaeengine.models.losses import LPIPSLoss

vae_model = "cross-attention/asymmetric-autoencoder-kl-x-2"
model = dict(
   type=AsymmetricAutoencoderKLModel,
   vae=dict(
      type=AsymmetricAutoencoderKL.from_pretrained,
      pretrained_model_name_or_path=vae_model),
   lpips_loss=dict(type=LPIPSLoss, loss_weight=0.1))
