#!/bin/bash

DATA_DIR=~/shared/hdd_ext/nvme1/public/vision/classification/imageNet

# 1) vit-s 1/2 (depth6 baseline) - 75.36%
torchrun --nproc_per_node=2 --master_port=12350 train_sh.py $DATA_DIR \
    --val-split val \
    --aa rand-m9-mstd0.5-inc1 --mixup .8 --cutmix 1.0 --aug-repeats 0 \
    --remode pixel --reprob 0.25 --drop-path .1 \
    --opt adamw --weight-decay .05 --sched cosine \
    --epochs 300 --lr 1e-3 --warmup-lr 1e-6 --min-lr 1e-5 --warmup-epochs 5 \
    --smoothing 0.1 --batch-size 256 --grad-accumulation 2 -j 8 --amp \
    --model vit_small_patch16_224_depth6 \
    --cuda 6,7

# # 2) ours (router partial_mlp) - 78.36%
# torchrun --nproc_per_node=2 --master_port=12350 train_sh.py $DATA_DIR \
#     --val-split val \
#     --aa rand-m9-mstd0.5-inc1 --mixup .8 --cutmix 1.0 --aug-repeats 0 \
#     --remode pixel --reprob 0.25 --drop-path .1 \
#     --opt adamw --weight-decay .05 --sched cosine \
#     --epochs 600 --lr 1e-3 --warmup-lr 1e-6 --min-lr 1e-5 --warmup-epochs 5 \
#     --smoothing 0.1 --batch-size 512 --grad-accumulation 2 -j 8 --amp \
#     --model router_vit_small_patch16_224_depth12_6_e256_partial_mlp \
#     --cuda 6,7
torchrun --nproc_per_node=2 --master_port=12350 train_sh.py $DATA_DIR \
    --val-split val \
    --aa rand-m9-mstd0.5-inc1 --mixup .8 --cutmix 1.0 --aug-repeats 0 \
    --remode pixel --reprob 0.25 --drop-path .1 \
    --opt adamw --weight-decay .05 --sched cosine \
    --epochs 600 --lr 1e-3 --warmup-lr 1e-6 --min-lr 1e-5 --warmup-epochs 5 \
    --smoothing 0.1 --batch-size 512 --grad-accumulation 2 -j 8 --amp \
    --model router_vit_small_patch16_224_depth12_6_e256_partial_mlp \
    --cuda 6,7 \
    --resume output/train/router_vit_small_patch16_224_depth12_6_e256_partial_mlp_20260323_151120/last.pth.tar

# 3) vit-s (depth12 baseline) - 79.80%
torchrun --nproc_per_node=2 --master_port=12350 train_sh.py $DATA_DIR \
    --val-split val \
    --aa rand-m9-mstd0.5-inc1 --mixup .8 --cutmix 1.0 --aug-repeats 0 \
    --remode pixel --reprob 0.25 --drop-path .1 \
    --opt adamw --weight-decay .05 --sched cosine \
    --epochs 300 --lr 1e-3 --warmup-lr 1e-6 --min-lr 1e-5 --warmup-epochs 5 \
    --smoothing 0.1 --batch-size 512 --grad-accumulation 2 -j 8 --amp \
    --model vit_small_patch16_224_depth12 \
    --cuda 7,8


# 4) new version
torchrun --nproc_per_node=2 --master_port=12350 train_sh.py $DATA_DIR \
    --val-split val \
    --aa rand-m9-mstd0.5-inc1 --mixup .8 --cutmix 1.0 --aug-repeats 0 \
    --remode pixel --reprob 0.25 --drop-path .1 \
    --opt adamw --weight-decay .05 --sched cosine \
    --epochs 600 --lr 1e-3 --warmup-lr 1e-6 --min-lr 1e-5 --warmup-epochs 5 \
    --smoothing 0.1 --batch-size 512 --grad-accumulation 2 -j 8 --amp \
    --model router_vit_small_patch16_224_depth12_pattern_router \
    --cuda 7,8



# validate 돌리는 방법
python validate.py \
  --data-dir "$DATA_DIR" \
  --model router_vit_small_patch16_224_depth12_pattern_router \
  --checkpoint ~/private/layer_router2/output/train/vit_small_patch16_224_depth12_20260405_014307 \
  --device cuda \
  --num-gpu 1 \
  --batch-size 128 