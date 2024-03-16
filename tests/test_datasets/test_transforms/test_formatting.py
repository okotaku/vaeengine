import copy
import unittest

import torch
from mmengine.registry import TRANSFORMS

from vaeengine.datasets import PackInputs


class TestPackInputs(unittest.TestCase):

    def test_transform(self):
        data = {"dummy": 1, "img": torch.zeros((3, 32, 32))}

        cfg = dict(type=PackInputs, input_keys=["img"])
        transform = TRANSFORMS.build(cfg)
        results = transform(copy.deepcopy(data))
        assert "inputs" in results

        assert "img" in results["inputs"]
        assert isinstance(results["inputs"]["img"], torch.Tensor)
        assert "dummy" not in results["inputs"]
