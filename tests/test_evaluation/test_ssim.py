from mmengine.testing import RunnerTestCase
from PIL import Image

from vaeengine.evaluation import SSIM


class TestSSIM(RunnerTestCase):

    def test_psnr_score(self):
        img = Image.open("tests/testdata/color.jpg")
        img2 = Image.open(
            "tests/testdata/dataset/cond.jpg").resize((img.width, img.height))
        psnr = SSIM()
        score = psnr(img, img)
        assert isinstance(score, float)
        assert score == 1.0

        score = psnr(img, img2)
        assert isinstance(score, float)
        assert score == 0.02995191703848259
