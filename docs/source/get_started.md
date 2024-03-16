# Development Environment Options

## Docker

Below are the quick steps for installing and running dreambooth training using Docker:

```bash
git clone https://github.com/okotaku/vaeengine
cd vaeengine
docker compose up -d
docker compose exec vaeengine vaeengine train autoencoderkl_sdv15_pokemon
```

## Devcontainer

You can also utilize the devcontainer to develop the VAEEngine. The devcontainer is a pre-configured development environment that runs in a Docker container. It includes all the necessary tools and dependencies for developing, building, and testing the VAEEngine.

1. Clone repository:

```
git clone https://github.com/okotaku/vaeengine
```

2. Open the cloned repository in Visual Studio Code.

3. Click on the "Reopen in Container" button located in the bottom right corner of the window. This action will open the repository within a devcontainer.

4. Run the following command to start training with the selected config:

```bash
vaeengine train autoencoderkl_sdv15_pokemon
```

# Get Started

vaeengine makes training easy through its pre-defined configs. These configs provide a streamlined way to start your training process. Here's how you can get started using one of the pre-defined configs:

1. **Choose a config**: You can find sample pre-defined configs in the [`configs`](vaeengine/configs/) directory of the vaeengine repository. For example, if you wish to train a AutoencoderKL model, you can use the [`configs/autoencoderkl/autoencoderkl_sdv15_pokemon.py`](vaeengine/configs/autoencoderkl/autoencoderkl_sdv15_pokemon.py).

2. **Start Training**: Open a terminal and run the following command to start training with the selected config:

```bash
vaeengine train autoencoderkl_sdv15_pokemon
```

3. **Monitor Progress and get results**: The training process will begin, and you can track its progress. The outputs of the training will be located in the `work_dirs/autoencoderkl_sdv15_pokemon` directory, specifically when using the `autoencoderkl_sdv15_pokemon` config.

```
work_dirs/autoencoderkl_sdv15_pokemon
├── 20230802_033741
|   ├── 20230802_033741.log  # log file
|   └── vis_data
|         ├── 20230802_033741.json  # log json file
|         ├── config.py  # config file for each experiment
|         └── vis_image  # visualized image from each step
├── step627/vae  # last step VAE model with diffusers format
|   ├── config.json  # conrfig file
|   └── diffusion_pytorch_model.bin  # weight for inferencing with diffusers.pipeline
├── epoch_1.pth  # checkpoint from each step
├── last_checkpoint  # last checkpoint, it can be used for resuming
├── scores.json  # lastest score
└── autoencoderkl_sdv15_pokemon.py  # latest config file
```
