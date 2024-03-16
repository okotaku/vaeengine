import copy

from mmengine.config import Config
from mmengine.testing import RunnerTestCase

from vaeengine.engine.hooks import InferHook


class TestVisualizationHook(RunnerTestCase):

    def test_before_train(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.model = Config.fromfile("tests/configs/vae.py").model
        runner = self.build_runner(cfg)

        hook = InferHook(
            dataset="tests/testdata/dataset",
            image_column="file_name",
            height=64,
            width=64)
        hook.before_train(runner)

    def test_after_train_epoch(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.model = Config.fromfile("tests/configs/vae.py").model

        # test epoch-based
        runner = self.build_runner(cfg)
        hook = InferHook(
            dataset="tests/testdata/dataset",
            image_column="file_name",
            height=64,
            width=64)
        hook.after_train_epoch(runner)

    def test_after_train_iter(self):
        cfg = copy.deepcopy(self.iter_based_cfg)
        cfg.train_cfg.max_iters = 100
        cfg.model = Config.fromfile("tests/configs/vae.py").model
        runner = self.build_runner(cfg)
        hook = InferHook(
            dataset="tests/testdata/dataset",
            image_column="file_name",
            by_epoch=False,
            height=64,
            width=64)
        for i in range(3):
            hook.after_train_iter(runner, i)
            runner.train_loop._iter += 1
