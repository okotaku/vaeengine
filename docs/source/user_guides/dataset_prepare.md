# Prepare Dataset

## Finetune 

1. Prepare `metadata.csv` and images.

The folder structure is

```
data/example
├── color.jpg
├── ...
└── metadata.csv
```

Example of `metadata.csv`.

```
file_name
color.jpg
```

2. Fix dataset config.

```
train_dataloader = dict(
    ...
    dataset=dict(
        ...
        dataset="data/example",
        image_column="file_name",
        csv="metadata.csv",
        ...
        )
    ...
)
```

3. Run training.
