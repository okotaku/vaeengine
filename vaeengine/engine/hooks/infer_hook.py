import os
from pathlib import Path
from typing import Optional

import mmengine
import numpy as np
from datasets import load_dataset
from mmengine.hooks import Hook
from mmengine.hooks.hook import DATA_BATCH
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner


class InferHook(Hook):
    """Basic hook that invoke visualizers and evaluator after train epoch.

    Args:
    ----
        dataset (str): Dataset name or path to dataset.
        csv (str): Caption csv file name when loading local folder.
            Defaults to 'metadata.csv'.
        interval (int): Visualization interval (every k iterations).
            Defaults to 1.
        by_epoch (bool): Whether to visualize by epoch. Defaults to True.
        height (int): The height in pixels of the generated image.
            Defaults to 512.
        width (int): The width in pixels of the generated image.
            Defaults to 512.

    """

    priority = "NORMAL"

    def __init__(self,
                 dataset: str,
                 csv: str = "metadata.csv",
                 interval: int = 1,
                 height: int = 512,
                 width: int = 512,
                 *,
                 by_epoch: bool = True,
                 **kwargs) -> None:
        self.kwargs = kwargs
        self.interval = interval
        self.by_epoch = by_epoch
        self.height = height
        self.width = width
        self.dataset_name = dataset

        if Path(dataset).exists():
            # load local folder
            data_file = os.path.join(dataset, csv)
            self.dataset = load_dataset(
                "csv", data_files=data_file)["train"]
        else:
            # load huggingface online
            self.dataset = load_dataset(dataset)["train"]

    def _visualize_and_eval(
        self, runner: Runner, step: int, suffix: str = "step") -> None:
        """Visualize and evaluate."""
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        images, ssim_scores, psnr_scores = model.infer(
            self.dataset,
            self.dataset_name,
            height=self.height,
            width=self.width,
            **self.kwargs)
        for i, image in enumerate(images):
            runner.visualizer.add_image(
                f"image{i}_{suffix}", image, step=step)

        # Evaluate
        ssim_text = f"SSIM is {np.mean(ssim_scores)}"
        runner.logger.info(ssim_text)
        psnr_text = f"PSNR is {np.mean(psnr_scores)}"
        runner.logger.info(psnr_text)
        mmengine.dump({
            "SSIM": np.mean(ssim_scores),
            "PSNR": np.mean(psnr_scores),
        }, f"{runner.work_dir}/scores.json")

    def before_train(self, runner: Runner) -> None:
        """Before train hook."""
        self._visualize_and_eval(runner, runner.iter, suffix="before_train")

    def after_train_iter(
            self,
            runner: Runner,
            batch_idx: int,
            data_batch: DATA_BATCH = None,  # noqa
            outputs: Optional[dict] = None) -> None:  # noqa
        """After train iter hook.

        Args:
        ----
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch.
            data_batch (DATA_BATCH, optional): The current data batch.
            outputs (dict, optional): The outputs of the current batch.

        """
        if self.by_epoch:
            return

        if self.every_n_inner_iters(batch_idx, self.interval):
            self._visualize_and_eval(runner, runner.iter)

    def after_train_epoch(self, runner: Runner) -> None:
        """After train epoch hook.

        Args:
        ----
            runner (Runner): The runner of the training process.

        """
        if self.by_epoch:
            self._visualize_and_eval(runner, runner.epoch)
