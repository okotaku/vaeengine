from unittest import TestCase

import numpy as np
import pytest
from mmengine.registry import TRANSFORMS

from vaeengine.datasets import MaskToTensor


class TestMaskToTensor(TestCase):

    def test_transform(self):
        data = {"mask": np.zeros((32, 32, 1))}

        # test transform
        trans = TRANSFORMS.build(dict(type=MaskToTensor))
        data = trans(data)
        assert data["mask"].shape == (1, 32, 32)

    def test_transform_list(self):
        data = {"mask": [np.zeros((32, 32, 1))] * 2}

        # test transform
        trans = TRANSFORMS.build(dict(type=MaskToTensor))
        with pytest.raises(
                AssertionError, match="MaskToTensor only support"):
            _ = trans(data)
