import numpy as np
from PIL import Image


class PSNR:
    """PSNR class for evaluating the similarity between two images."""

    def __call__(self, img1: Image.Image, img2: Image.Image) -> float:
        """Call function for PSNR."""
        mse = np.mean((np.array(img1) - np.array(img2)) ** 2)
        if mse == 0:
            return float("inf")
        return 20 * np.log10(255. / np.sqrt(mse))
