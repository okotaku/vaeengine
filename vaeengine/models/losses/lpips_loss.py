import lpips
import torch
import torch.nn.functional as F  # noqa

from vaeengine.models.losses.base import BaseLoss


class LPIPSLoss(BaseLoss):
    """LPIPS loss.

    Args:
    ----
        loss_weight (float, optional): Weight of this loss item.
            Defaults to ``1.``.
        reduction: (str): The reduction method for the loss.
            Defaults to 'mean'.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'lpips'.

    """

    def __init__(self,
                 loss_weight: float = 1.0,
                 loss_name: str = "lpips") -> None:

        super().__init__()
        self.loss_weight = loss_weight
        self._loss_name = loss_name

        self.lpips = lpips.LPIPS(net="alex")

    def forward(self,
                pred: torch.Tensor,
                gt: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
        ----
            pred (torch.Tensor): The predicted tensor.
            gt (torch.Tensor): The ground truth tensor.

        Returns:
        -------
            torch.Tensor: loss

        """
        return self.lpips(pred, gt).mean() * self.loss_weight