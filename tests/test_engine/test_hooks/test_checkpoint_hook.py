import copy
import shutil
from pathlib import Path

from diffusers import AutoencoderKL
from mmengine.model import BaseModel
from mmengine.registry import MODELS
from mmengine.testing import RunnerTestCase
from mmengine.testing.runner_test_case import ToyModel
from torch import nn

from vaeengine.engine.hooks import CheckpointHook


class DummyWrapper(BaseModel):

    def __init__(self, model) -> None:
        super().__init__()
        if not isinstance(model, nn.Module):
            model = MODELS.build(model)
        self.module = model

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class ToyModel2(ToyModel):

    def __init__(self) -> None:
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(
            "diffusers/tiny-stable-diffusion-torch", subfolder="vae")

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


class TestCheckpointHook(RunnerTestCase):

    def setUp(self) -> None:
        MODELS.register_module(name="DummyWrapper", module=DummyWrapper)
        MODELS.register_module(name="ToyModel2", module=ToyModel2)
        return super().setUp()

    def tearDown(self):
        MODELS.module_dict.pop("DummyWrapper")
        MODELS.module_dict.pop("ToyModel2")
        return super().tearDown()

    def test_init(self):
        CheckpointHook()

    def test_after_run(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.model.type = "ToyModel2"
        runner = self.build_runner(cfg)
        hook = CheckpointHook()
        hook.after_run(runner)

        assert (Path(runner.work_dir) / (f"step{runner.iter}/vae/"
                     "diffusion_pytorch_model.safetensors")).exists()
        shutil.rmtree(
            Path(runner.work_dir) / f"step{runner.iter}")
