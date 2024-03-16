import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity


class SSIM:
    """SSIM class for evaluating the similarity between two images."""

    def __call__(self, img1: Image.Image, img2: Image.Image) -> float:
        """Call function for SSIM."""
        img1 = np.array(img1)
        img2 = np.array(img2)
        ssims = [
            structural_similarity(
                img1[..., i], img2[..., i], full=True)[0] for i in range(img1.shape[2])]
        return np.mean(ssims)
