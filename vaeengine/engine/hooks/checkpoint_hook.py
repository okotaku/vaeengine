import os.path as osp

import torch
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner


class CheckpointHook(Hook):
    """Delete 'vae' from checkpoint for efficient save."""

    priority = "VERY_LOW"

    def after_run(self, runner: Runner) -> None:
        """After run hook."""
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        ckpt_path = osp.join(runner.work_dir, f"step{runner.iter}")
        if hasattr(model, "vae"):
            for p in model.vae.parameters():
                is_contiguous = p.is_contiguous()
                break
            if not is_contiguous:
                model.vae = model.vae.to(
                    memory_format=torch.contiguous_format)
            model.vae.save_pretrained(osp.join(ckpt_path, "vae"))
