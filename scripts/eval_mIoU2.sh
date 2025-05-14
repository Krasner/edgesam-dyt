#!/bin/bash

# ./scripts/eval_mIoU2.sh "/home/ubuntu/edgesam-dyt/training/configs/rep_vit_m1_dyt_fuse_enc_dec_4m_ft_bp_iter2b_sa_distill.yaml" "/home/ubuntu/edgesam-dyt/output/rep_vit_m1_dyt_fuse_enc_dec_4m_ft_bp_iter2b_sa_distill/default/ckpt_epoch_39.pth" --local_rank 0

all_args=("$@")
rest_args=("${all_args[@]:2}")

CUDA_VISIBLE_DEVICES=0 PYTHONPATH="/home/ubuntu/edgesam-dyt/" python -m torch.distributed.launch --master_port 29501 --nproc_per_node 1 \
    evaluation/eval_mIoU.py --launcher pytorch \
    $1 \
    --checkpoint $2 \
    --num-samples -1 \
    --max-prompt-bs -1 \
    --img-bs 1 \
    --dataset 'coco' \
    --refine-iter 3 \
    ${rest_args[@]}

CUDA_VISIBLE_DEVICES=0 PYTHONPATH="/home/ubuntu/edgesam-dyt/" python -m torch.distributed.launch --master_port 29501 --nproc_per_node 1 \
    evaluation/eval_mIoU.py --launcher pytorch \
    $1 \
    --checkpoint $2 \
    --num-samples -1 \
    --max-prompt-bs -1 \
    --img-bs 1 \
    --prompt-types 'point' \
    --dataset 'coco' \
    --point-from 'mask-center' \
    --refine-iter 3 \
    ${rest_args[@]}