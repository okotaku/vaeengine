import torch
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner


class CompileHook(Hook):
    """Compile Hook.

    Args:
    ----
        backend (str): The backend to use for compilation.
            Defaults to "inductor".
        mode (str): The mode to use for compilation. Defaults to None.

    """

    priority = "VERY_LOW"

    def __init__(self, backend: str = "inductor", mode: str | None = None,
                 ) -> None:
        super().__init__()
        self.backend = backend
        self.mode = mode

    def before_train(self, runner: Runner) -> None:
        """Compile the model.

        Args:
        ----
            runner (Runner): The runner of the training process.

        """
        model = runner.model
        if is_model_wrapper(model):
            model = model.module

        if hasattr(model, "vae"):
            model.vae = torch.compile(
                model.vae, backend=self.backend, mode=self.mode)
