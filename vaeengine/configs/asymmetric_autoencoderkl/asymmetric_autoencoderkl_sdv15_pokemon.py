from mmengine.config import read_base

with read_base():
    from .._base_.datasets.pokemon_baseline_mask import *
    from .._base_.default_runtime import *
    from .._base_.models.asymmetric_autoencoderkl_sdv15 import *
    from .._base_.schedules.autoencoder_50e_baseline import *
