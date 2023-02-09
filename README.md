# NeRF-pytorch

Adapted from [NeRF-pytorch](https://github.com/yenchenlin/nerf-pytorch), re-structured for better interpretability and extensibility.


## Train

```shell script
python train_nerf.py config/blender/lego_relu.yaml --use_wandb
python train_nerf.py config/llff/fern.yaml --use_wandb
```

## Test

```shell script
python test_nerf.py config/blender/lego_relu.yaml --checkpoint 199000
python test_nerf.py config/llff/fern.yaml --checkpoint 199000
```