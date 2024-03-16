import numpy as np
from mmengine.testing import RunnerTestCase
from PIL import Image

from vaeengine.evaluation import PSNR


class TestPSNR(RunnerTestCase):

    def test_psnr_score(self):
        img = Image.open("tests/testdata/color.jpg")
        img2 = Image.open(
            "tests/testdata/dataset/cond.jpg").resize((img.width, img.height))
        psnr = PSNR()
        score = psnr(img, img)
        assert isinstance(score, float)
        assert score == np.inf

        score = psnr(img, img2)
        assert isinstance(score, float)
        assert score == 27.890020066842215
