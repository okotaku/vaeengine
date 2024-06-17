import torch

from vaeengine.datasets.transforms.base import BaseTransform


class MaskToTensor(BaseTransform):
    """MaskToTensor.

    1. Convert mask to tensor.
    2. Transpose mask from (H, W, 1) to (1, H, W)

    Args:
    ----
        key (str): `key` to apply augmentation from results.
            Defaults to 'mask'.

    """

    def __init__(self, key: str = "mask") -> None:
        self.key = key

    def transform(self, results: dict) -> dict | tuple[list, list] | None:
        """Transform.

        Args:
        ----
            results (dict): The result dict.

        """
        assert not isinstance(results[self.key], list), (
            "MaskToTensor only support single image.")
        # (1, 3, 224, 224) -> (3, 224, 224)
        results[self.key] = torch.Tensor(results[self.key]).permute(2, 0, 1)
        return results
