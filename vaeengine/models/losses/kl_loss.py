import torch
import torch.nn.functional as F  # noqa

from vaeengine.models.losses.base import BaseLoss


class KLLoss(BaseLoss):
    """KL loss.

    Args:
    ----
        loss_weight (float, optional): Weight of this loss item.
            Defaults to ``1.``.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'kl'.

    """

    def __init__(self,
                 loss_weight: float = 1.0,
                 loss_name: str = "kl") -> None:

        super().__init__()
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(self,
                posterior: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
        ----
            posterior (torch.Tensor): The posterior tensor.

        Returns:
        -------
            torch.Tensor: loss

        """
        return posterior.kl().mean() * self.loss_weight
