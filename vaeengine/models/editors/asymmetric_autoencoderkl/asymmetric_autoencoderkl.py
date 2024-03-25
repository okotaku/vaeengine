from torch import nn

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
