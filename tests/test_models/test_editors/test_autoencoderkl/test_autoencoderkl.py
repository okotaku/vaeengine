from unittest import TestCase

import pytest
import torch
from datasets import load_dataset
from diffusers import AutoencoderKL
from mmengine.optim import OptimWrapper
from mmengine.registry import MODELS
from torch.optim import SGD

from vaeengine.models.editors import AutoencoderKLModel


class TestStableDiffusion(TestCase):

    def _get_config(self) -> dict:
        base_model = "diffusers/tiny-stable-diffusion-torch"
        return dict(
            type=AutoencoderKLModel,
             vae=dict(
                type=AutoencoderKL.from_pretrained,
                pretrained_model_name_or_path=base_model,
                subfolder="vae"))

    def test_infer(self):
        cfg = self._get_config()
        model =  MODELS.build(cfg)

        # test infer
        dataset = load_dataset(
                "csv", data_files="tests/testdata/dataset/metadata.csv")["train"]
        result, ssim_scores, psnr_scores = model.infer(
            dataset,
            "tests/testdata/dataset",
            image_column="file_name",
            height=64,
            width=64)
        assert len(result) == 1
        assert result[0].shape == (64, 64, 3)
        assert 0 <= ssim_scores[0] <= 1
        assert psnr_scores[0] >= 0

        # test device
        assert model.device.type == "cpu"

    def test_train_step(self):
        # test load with loss module
        cfg = self._get_config()
        model =  MODELS.build(cfg)

        # test train step
        data = dict(
            inputs=dict(img=[torch.zeros((3, 64, 64))]))
        optimizer = SGD(model.parameters(), lr=0.1)
        optim_wrapper = OptimWrapper(optimizer)
        log_vars = model.train_step(data, optim_wrapper)
        assert log_vars
        assert isinstance(log_vars["loss"], torch.Tensor)

    def test_train_step_with_gradient_checkpointing(self):
        # test load with loss module
        cfg = self._get_config()
        cfg.update(gradient_checkpointing=True)
        model =  MODELS.build(cfg)

        # test train step
        data = dict(
            inputs=dict(img=[torch.zeros((3, 64, 64))]))
        optimizer = SGD(model.parameters(), lr=0.1)
        optim_wrapper = OptimWrapper(optimizer)
        log_vars = model.train_step(data, optim_wrapper)
        assert log_vars
        assert isinstance(log_vars["loss"], torch.Tensor)

    def test_val_and_test_step(self):
        cfg = self._get_config()
        model =  MODELS.build(cfg)

        # test val_step
        with pytest.raises(NotImplementedError, match="val_step is not"):
            model.val_step(torch.zeros((1, )))

        # test test_step
        with pytest.raises(NotImplementedError, match="test_step is not"):
            model.test_step(torch.zeros((1, )))
