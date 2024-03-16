import pytest
import torch
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from PIL import Image

from vaeengine.models.losses import KLLoss, L2Loss, LPIPSLoss


def test_l2_loss():
    with pytest.raises(
            AssertionError, match="reduction should be 'mean' or 'none'"):
        _ = L2Loss(reduction="dummy")

    pred = torch.Tensor([[5, -5, 0], [5, -5, 0]])
    gt = torch.Tensor([[1, 0, 1], [0, 1, 0]])
    weight = torch.Tensor([[1], [0.1]])

    loss = L2Loss()
    assert torch.allclose(loss(pred, gt), torch.tensor(17.1667))
    assert torch.allclose(loss(pred, gt, weight=weight), torch.tensor(8.0167))

    loss = L2Loss(reduction="none")
    assert loss(pred, gt).shape == (2, 3)


def test_kl_loss():
    img = Image.open("tests/testdata/color.jpg")
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path="diffusers/tiny-stable-diffusion-torch",
        subfolder="vae")
    vae_scale_factor = 2**(len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
    posterior = vae.encode(image_processor.preprocess(img)).latent_dist

    loss = KLLoss()
    assert torch.allclose(loss(posterior), torch.tensor(7925.0908))


def test_lpips_loss():
    # test asymmetric_loss
    img = Image.open("tests/testdata/color.jpg")
    img2 = Image.open("tests/testdata/dataset/cond.jpg",
                      ).resize((img.width, img.height))

    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path="diffusers/tiny-stable-diffusion-torch",
        subfolder="vae")
    vae_scale_factor = 2**(len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
    img = image_processor.preprocess(img)
    img2 = image_processor.preprocess(img2)

    loss = LPIPSLoss()
    assert torch.allclose(loss(img, img), torch.tensor(0.0))

    assert torch.allclose(loss(img, img2), torch.tensor(0.8807),
                          rtol=1e-3, atol=1e-4)
