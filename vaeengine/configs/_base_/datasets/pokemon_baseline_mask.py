import torchvision
from mmengine.dataset import DefaultSampler

from vaeengine.datasets import HFDataset
from vaeengine.datasets.transforms import (
    DumpImage,
    LoadMask,
    MaskToTensor,
    PackInputs,
    RandomChoice,
    RandomCrop,
    RandomHorizontalFlip,
    TorchVisonTransformWrapper,
)
from vaeengine.engine.hooks import CheckpointHook, InferHook

train_pipeline = [
    dict(type=TorchVisonTransformWrapper,
         transform=torchvision.transforms.Resize,
         size=512, interpolation="bilinear"),
    dict(type=RandomCrop, size=512),
    dict(type=RandomHorizontalFlip, p=0.5),
    dict(type=RandomChoice,
         transforms=[
            [dict(type=RandomChoice,
                transforms=[
                    [dict(
                        type=LoadMask,
                        mask_mode="irregular",
                        mask_config=dict(
                            num_vertices=(4, 10),
                            max_angle=6.0,
                            length_range=(20, 200),
                            brush_width=(10, 100),
                            area_ratio_range=(0.15, 0.65)))],
                    [dict(
                        type=LoadMask,
                        mask_mode="irregular",
                        mask_config=dict(
                            num_vertices=(1, 5),
                            max_angle=6.0,
                            length_range=(40, 450),
                            brush_width=(20, 250),
                            area_ratio_range=(0.15, 0.65)))],
                    [dict(
                        type=LoadMask,
                        mask_mode="irregular",
                        mask_config=dict(
                            num_vertices=(4, 70),
                            max_angle=6.0,
                            length_range=(15, 100),
                            brush_width=(5, 20),
                            area_ratio_range=(0.15, 0.65)))],
                    [dict(
                        type=LoadMask,
                        mask_mode="bbox",
                        mask_config=dict(
                            max_bbox_shape=(150, 150),
                            max_bbox_delta=50,
                            min_margin=0))],
                    [dict(
                        type=LoadMask,
                        mask_mode="bbox",
                        mask_config=dict(
                            max_bbox_shape=(300, 300),
                            max_bbox_delta=100,
                            min_margin=10))],
                ])],
         [dict(
                        type=LoadMask,
                        mask_mode="whole")]],
         prob=[0.9, 0.1],
    ),
    dict(type=TorchVisonTransformWrapper,
         transform=torchvision.transforms.ToTensor),
    dict(type=MaskToTensor),
    dict(type=DumpImage, max_imgs=10, dump_dir="work_dirs/dump"),
    dict(type=TorchVisonTransformWrapper,
         transform=torchvision.transforms.Normalize, mean=[0.5], std=[0.5]),
    dict(type=PackInputs,
        input_keys=["img", "mask"]),
]
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=dict(
        type=HFDataset,
        dataset="diffusers/pokemon-gpt4-captions",
        pipeline=train_pipeline),
    sampler=dict(type=DefaultSampler, shuffle=True),
)

val_dataloader = None
val_evaluator = None
test_dataloader = val_dataloader
test_evaluator = val_evaluator

custom_hooks = [
    dict(type=InferHook, dataset="diffusers/pokemon-gpt4-captions"),
    dict(type=CheckpointHook),
]
