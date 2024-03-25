from pathlib import Path

import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import load_image
from PIL import Image
from torch import nn
from torch.utils.data import Dataset

from vaeengine.models.editors.asymmetric_autoencoderkl.data_preprocessor import (
    AsymmetricDataPreprocessor,
)
from vaeengine.models.editors.autoencoderkl import AutoencoderKLModel


class AsymmetricAutoencoderKLModel(AutoencoderKLModel):
    """Asymmetric AutoencoderKL Model.

    Args:
    ----
        data_preprocessor (dict, optional): The pre-process config of
            :class:`AsymmetricDataPreprocessor`.

    """

    def __init__(self,
                 *args,
                 data_preprocessor: dict | nn.Module | None = None,
                 **kwargs) -> None:
        if data_preprocessor is None:
            data_preprocessor = {"type": AsymmetricDataPreprocessor}

        super().__init__(
            *args,
            data_preprocessor=data_preprocessor,
            **kwargs)  # type: ignore[misc]

    @torch.no_grad()
    def infer(
        self,
        dataset: Dataset,
        dataset_name: str,
        image_column: str = "image",
        mask_column: str = "mask",
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
            mask_column (str): Mask column name. Defaults to 'mask'.
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
        for i, d in enumerate(dataset):
            img = d[image_column]
            if isinstance(img, str):
                img = str(Path(dataset_name) / img)
            pil_img = load_image(img) if isinstance(img, str) else img
            pil_img = pil_img.convert("RGB").resize(
                (width, height), Image.BILINEAR)
            mask = d[mask_column] if mask_column in d else Image.open(
                f"work_dirs/masks/mask_{i}.png").convert("L").resize(
                (width, height), Image.BILINEAR)
            mask = np.array(mask) / 255
            mask = torch.Tensor(mask).unsqueeze(0).permute(
                2, 0, 1).to(self.weight_dtype).to(self.device)
            pixel_values = image_processor.preprocess(
                pil_img).to(self.weight_dtype).to(self.device)

            latents = self.vae.encode(pixel_values).latent_dist.sample()
            image = self.vae.decode(latents, pixel_values,
                                    mask, return_dict=False)[0]
            image = image_processor.postprocess(image.detach(), output_type="pil")[0]
            ssim_scores.append(self.ssim(image, pil_img))
            psnr_scores.append(self.psnr(image, pil_img))
            max_save_img = 30
            if len(images) > max_save_img:
                # save only 30 images
                continue
            images.append(np.array(image))

        return images, ssim_scores, psnr_scores

    def forward(
            self,
            inputs: dict,
            data_samples: list | None = None,  # noqa
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
        z = posterior.sample() if self.sample_posterior else posterior.mode()
        model_pred = self.vae.decode(z,
                                     inputs["img"].to(self.weight_dtype),
                                     inputs["mask"].to(self.weight_dtype),
                                     ).sample

        return self.loss(model_pred, posterior, inputs["img"])
