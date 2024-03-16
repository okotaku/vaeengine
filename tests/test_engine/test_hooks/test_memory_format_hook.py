import copy
import os

import pytest
from mmengine.config import Config
from mmengine.testing import RunnerTestCase

from vaeengine.engine.hooks import MemoryFormatHook


@pytest.mark.skipif("GITHUB_ACTION" in os.environ,
                    reason="skip external api call during CI")
class TestMemoryFormatHook(RunnerTestCase):

    def test_init(self) -> None:
        MemoryFormatHook()

    def test_before_train(self) -> None:
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.model = Config.fromfile("tests/configs/vae.py").model
        runner = self.build_runner(cfg)
        hook = MemoryFormatHook()
        for p in runner.model.vae.parameters():
            is_contiguous = p.is_contiguous()
            break
        assert is_contiguous

        # run hook
        hook.before_train(runner)

        for p in runner.model.vae.parameters():
            is_contiguous = p.is_contiguous()
            break
        assert not is_contiguous
