# experimental results of the layer router method

### Experimental results
|   Model    |   layers    |     block     |  Top-1 Acc.   |
|:----------:|:-----------:|:-------------:|:-------------:|
|  ResNet50  |  [1,2,3,1]  |  Bottlenceck  |     74.76     |

Training script for early_resnet50 (RN50 with layers(1,2,3,1)) 
```
CUDA_VISIBLE_DEVICES=2,3,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port=12346 train.py \
../DATA/imageNet --model early_resnet50 --mean 0.485 0.456 0.406 --std 0.229 0.224 0.225 --aa rand-m7-mstd0.5-inc1 \
--mixup .1 --cutmix 1.0 --aug-repeats 3 --remode pixel --reprob --crop-pct 0.95 --drop-path .05 --smoothing 0.0 \
--bce-loss --bce-target-thresh 0.2 --opt lamb --weight-decay .02 --sched cosine --epochs 300 --lr 2.5e-3 \
--warmup-lr 1e-4 -b 256 -j 16 --amp --channels-last --log-wandb --output ./early_resnet50
```

|       Model       |  depth  | embed_dims | num_heads | gap |  Top-1 Acc.  |
|:-----------------:|:-------:|:----------:|:---------:|:---:|:------------:|
|  ViT-tiny/16@224  |    6    |    192     |     3     |  O  |    63.23     |

Training script for early_vit_tiny_patch16_224_gap (ViT-tiny/16 with depth 6)
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=12345 train.py \
../DATA/imageNet --model early_vit_tiny_patch16_224_gap --mean 0.485 0.456 0.406 --std 0.229 0.224 0.225 \
--aa rand-m9-mstd0.5-inc1 --mixup 0.8 --cutmix 1.0 --aug-repeats 0 --remode pixel --reprob 0.25 --batch-size 256 \
--epochs 300 --input-size 3 224 224 --drop-path 0.1 --opt adamw --opt-eps 1e-8 --momentum 0.9 --weight-decay 0.05 \
--sched cosine --lr 5e-4 --warmup-lr 1e-6 --min-lr 1e-5 --decay-epochs 30 --warmup-epochs 5 --color-jitter 0.3 \
--smoothing 0.1 --train-interpolation bicubic --amp --channels-last --log-wandb --output ./early_vit_tiny_gap
```

Training script for baseline (ViT-small with depth 6)
```
torchrun --nproc_per_node=2 --master_port=12348 train_sh.py /home/data/imagenet --aa rand-m9-mstd0.5-inc1 --mixup .8 --cutmix 1.0 --aug-repeats 0 --remode pixel --reprob 0.25 --drop-path .1 --opt adamw --weight-decay .05 --sched cosine --epochs 300 --lr 1e-3 --warmup-lr 1e-6 --min-lr 1e-5 --warmup-epoch 5 --smoothing 0.1 --batch-size 512 --grad-accumulation 2 -j 8 --amp --model vit_small_patch16_224_depth6 --cuda 0,1
```

Training script for ours (ViT-small with depth 6/12)
```
torchrun --nproc_per_node=2 --master_port=12348 train_sh.py /home/data/imagenet --aa rand-m9-mstd0.5-inc1 --mixup .8 --cutmix 1.0 --aug-repeats 0 --remode pixel --reprob 0.25 --drop-path .1 --opt adamw --weight-decay .05 --sched cosine --epochs 600 --lr 1e-3 --warmup-lr 1e-6 --min-lr 1e-5 --warmup-epoch 5 --smoothing 0.1 --batch-size 512 --grad-accumulation 2 -j 8 --amp --model router_vit_small_patch16_224_depth12_6_e256 --cuda 0,1
```
|       Model    |  Top-1 Acc.  |
|:-----------------:|:-------:|
|  baseline |    75.2     |
|  ours |    76.6     |


