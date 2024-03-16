import copy

from diffusers import AutoencoderKL
from mmengine.config import Config
from mmengine.testing import RunnerTestCase

from vaeengine.engine.hooks import CompileHook


class TestCompileHook(RunnerTestCase):

    def test_init(self) -> None:
        CompileHook()

    def test_before_train(self) -> None:
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.model = Config.fromfile("tests/configs/vae.py").model
        runner = self.build_runner(cfg)
        hook = CompileHook()
        assert isinstance(runner.model.vae, AutoencoderKL)
        # compile
        hook.before_train(runner)
        assert not isinstance(runner.model.vae, AutoencoderKL)
