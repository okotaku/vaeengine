from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import load_image
from mmengine.model import BaseModel
from mmengine.registry import MODELS
from PIL import Image
from torch import nn
from torch.utils.data import Dataset

from vaeengine.evaluation import PSNR, SSIM
from vaeengine.models.editors.autoencoderkl.data_preprocessor import (
    DataPreprocessor,
)
from vaeengine.models.losses import KLLoss, L2Loss, LPIPSLoss

weight_dtype_dict = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

class AutoencoderKLModel(BaseModel):
    """AutoencoderKL.

    Args:
    ----
        vae (dict): Config of vae.
        loss (dict): Config of loss. Defaults to
            ``dict(type='L2Loss', loss_weight=1.0)``.
        kl_loss (dict): Config of kl_loss. Defaults to None.
        lpips_loss (dict): Config of lpips_loss. Defaults to
            ``dict(type='LPIPSLoss', loss_weight=1e-1)``.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`DataPreprocessor`.
        weight_dtype (str): The weight dtype. Choose from "fp32", "fp16" or
            "bf16".  Defaults to 'fp32'.
        freeze_encoder (bool): Whether to freeze the encoder. Defaults to True.
        gradient_checkpointing (bool): Whether to use gradient checkpointing.
            Defaults to False.

    """

    def __init__(
        self,
        vae: dict,
        loss: dict | None = None,
        kl_loss: dict | None = None,
        lpips_loss: dict | None = None,
        data_preprocessor: dict | nn.Module | None = None,
        weight_dtype: str = "fp32",
        *,
        freeze_encoder: bool = True,
        gradient_checkpointing: bool = False,
    ) -> None:
        if data_preprocessor is None:
            data_preprocessor = {"type": DataPreprocessor}
        if loss is None:
            loss = {}
        if lpips_loss is None:
            lpips_loss = {}
        super().__init__(data_preprocessor=data_preprocessor)
        self.weight_dtype = weight_dtype_dict[weight_dtype]

        if not isinstance(loss, nn.Module):
            loss = MODELS.build(
                loss,
                default_args={"type": L2Loss, "loss_weight": 1.0})
        self.loss_module: nn.Module = loss

        if not isinstance(kl_loss, nn.Module) and kl_loss is not None:
            kl_loss = MODELS.build(
                kl_loss,
                default_args={"type": KLLoss, "loss_weight": 1e-6})
        self.kl_loss_module: nn.Module | None = kl_loss

        if not isinstance(lpips_loss, nn.Module):
            lpips_loss = MODELS.build(
                lpips_loss,
                default_args={"type": LPIPSLoss, "loss_weight": 1e-1})
        self.lpips_loss_module: nn.Module = lpips_loss

        self.vae = MODELS.build(vae)
        if freeze_encoder:
            self.vae.encoder.requires_grad_(requires_grad=False)
            self.vae.quant_conv.requires_grad_(requires_grad=False)
        if gradient_checkpointing:
            self.vae.enable_gradient_checkpointing()

        if self.weight_dtype != torch.float32:
            self.to(self.weight_dtype)

        self.ssim = SSIM()
        self.psnr = PSNR()

    @property
    def device(self) -> torch.device:
        """Get device information.

        Returns
        -------
            torch.device: device.

        """
        return next(self.parameters()).device

    @torch.no_grad()
    def infer(
        self,
        dataset: Dataset,
        dataset_name: str,
        image_column: str = "image",
        height: int = 512,
        width: int = 512,
        **kwargs,  # noqa: ARG002
        ) -> tuple[list[np.ndarray], list[float], list[float]]:
        """Inference function.

        Args:
        ----
            dataset (Dataset): The input dataset.
            dataset_name (str): The dataset name.
            image (`List[Union[str, Image.Image]]`):
                The input images.
            image_column (str): Image column name. Defaults to 'image'.
            height (int):
                The height in pixels of the generated image. Defaults to 512.
            width (int):
                The width in pixels of the generated image. Defaults to 512.
            **kwargs: Other arguments.

        """
        vae_scale_factor = 2**(len(self.vae.config.block_out_channels) - 1)
        image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
        images: list = []
        ssim_scores: list = []
        psnr_scores: list = []
        for d in dataset:
            img = d[image_column]
            if isinstance(img, str):
                img = str(Path(dataset_name) / img)
            pil_img = load_image(img) if isinstance(img, str) else img
            pil_img = pil_img.convert("RGB").resize(
                (width, height), Image.BILINEAR)
            pixel_values = image_processor.preprocess(
                pil_img).to(self.weight_dtype).to(self.device)

            latents = self.vae.encode(pixel_values).latent_dist.sample()
            image = self.vae.decode(latents, return_dict=False)[0]
            image = image_processor.postprocess(image.detach(), output_type="pil")[0]
            ssim_scores.append(self.ssim(image, pil_img))
            psnr_scores.append(self.psnr(image, pil_img))
            max_save_img = 30
            if len(images) > max_save_img:
                # save only 30 images
                continue
            images.append(np.array(image))

        return images, ssim_scores, psnr_scores

    def val_step(
            self,
            data: Union[tuple, dict, list]  # noqa
    ) -> list:
        """Val step."""
        msg = "val_step is not implemented now, please use infer."
        raise NotImplementedError(msg)

    def test_step(
            self,
            data: Union[tuple, dict, list]  # noqa
    ) -> list:
        """Test step."""
        msg = "test_step is not implemented now, please use infer."
        raise NotImplementedError(msg)

    def loss(self,
             model_pred: torch.Tensor,
             posterior: torch.Tensor,
             gt: torch.Tensor) -> dict[str, torch.Tensor]:
        """Calculate loss."""
        loss_dict = {}
        # calculate loss in FP32
        loss = self.loss_module(
            model_pred.float(), gt.float())
        lpips_loss = self.lpips_loss_module(
            model_pred.float(), gt.float())
        loss_dict["mse_loss"] = loss
        loss_dict["lpips_loss"] = lpips_loss
        if self.kl_loss_module is not None:
            kl_loss = self.kl_loss_module(posterior)
            loss_dict["kl_loss"] = kl_loss
        return loss_dict

    def forward(
            self,
            inputs: dict,
            data_samples: Optional[list] = None,  # noqa
            mode: str = "loss") -> dict:
        """Forward function.

        Args:
        ----
            inputs (dict): The input dict.
            data_samples (Optional[list], optional): The data samples.
                Defaults to None.
            mode (str, optional): The mode. Defaults to "loss".

        Returns:
        -------
            dict: The loss dict.

        """
        assert mode == "loss"
        posterior = self.vae.encode(inputs["img"].to(self.weight_dtype)).latent_dist
        z = posterior.mode()
        model_pred = self.vae.decode(z).sample

        return self.loss(model_pred, posterior, inputs["img"])
