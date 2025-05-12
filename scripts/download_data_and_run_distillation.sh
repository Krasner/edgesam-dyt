#!/bin/bash

# download files
# links can be found here: https://scontent-lga3-3.xx.fbcdn.net/m1/v/t6/An8MNcSV8eixKBYJ2kyw6sfPh-J9U4tH2BV7uPzibNa0pu4uHi6fyXdlbADVO4nfvsWpTwR8B0usCARHTz33cBQNrC0kWZsD1MbBWjw.txt?_nc_oc=AdgcU7b6EMb8dmnUOmBnNcphAQtGJcPBsftVcyHLozQOC7wsCRN9ahWKVnaCixV-YOaK5N1cpJrDD0kbn09POlFI&ccb=10-5&oh=00_AYGxM1hEmsynb7Lm6BLn996srLGL-JOKxT4c7VBLagRFuw&oe=67FDCA98&_nc_sid=0fdd51
cd /home/ubuntu/edgesam-dyt/datasets/
mkdir train
mkdir val

while read file_name cdn_link; do aria2c -x4 -c -o "$file_name" "$cdn_link"; done < ../training/sa_train_subset.txt &&

mv sa_*.tar ./train/ &&

while read file_name cdn_link; do aria2c -x4 -c -o "$file_name" "$cdn_link"; done < ../training/sa_val_subset.txt &&

mv sa_*.tar ./val/ &&

cd ./train
for file in *.tar; do tar xvf "${file}" && rm "${file}"; done

cd ../val
for file in *.tar; do tar xvf "${file}" && rm "${file}"; done

cd ..
find train/ -name "*.jpg" -exec mv {} SA-1B/images/train/ \;
find train/ -name "*.json" -exec mv {} SA-1B/annotations/train/ \;
find val/ -name "*.jpg" -exec mv {} SA-1B/images/val/ \;
find val/ -name "*.json" -exec mv {} SA-1B/annotations/val/ \;

# download necessary weight files
cd ..
mkdir weights && cd weights
wget https://github.com/THU-MIG/RepViT/releases/download/v1.0/repvit_m0_9_distill_450e.pth
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
cd ../scripts/
python extract_weights.py

cd ..
# pre-compute embeddings
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="/home/ubuntu/edgesam-dyt/" python -m torch.distributed.launch --master_port 29501 --nproc_per_node 1 training/save_embedding.py --cfg training/configs/teacher/sam_vit_huge_sa1b.yaml --batch-size 32 --eval --resume weights/sam_vit_h_4b8939.pth

# STEP 1: encoder distillation
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="/home/ubuntu/edgesam-dyt/" python -m torch.distributed.launch --master_port 29501 --nproc_per_node 1 training/train.py --cfg training/configs/rep_vit_m1_dyt_fuse_sa_distill.yaml --output ./output/ --batch-size 96

# STEP 2: encoder + frozen mask decoder (same as in sam) distillation
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="/home/ubuntu/edgesam-dyt/" python -m torch.distributed.launch --master_port 29501 --nproc_per_node 1 training/train.py --cfg training/configs/rep_vit_m1_dyt_fuse_enc_dec_4m_ft_bp_iter2b_sa_distill_frozen_mask.yaml --output ./output/ --batch-size 16

# STEP 3: frozen encoder + mask decoder (layernorms replaced with dyt layers, gelu tanh approx) distillation
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="/home/ubuntu/edgesam-dyt/" python -m torch.distributed.launch --master_port 29501 --nproc_per_node 1 training/train.py --cfg training/configs/rep_vit_m1_dyt_fuse_enc_dec_4m_ft_bp_iter2b_sa_distill.yaml --output ./output/ --batch-size 16

# export onnx
PYTHONPATH="/home/ubuntu/edgesam-dyt/" python scripts/export_onnx_model.py "/home/ubuntu/edgesam-dyt/output/rep_vit_m1_dyt_fuse_enc_dec_4m_ft_bp_iter2b_sa_distill/default/ckpt_epoch_39.pth" --simplify
PYTHONPATH="/home/ubuntu/edgesam-dyt/" python scripts/export_onnx_model.py "/home/ubuntu/edgesam-dyt/output/rep_vit_m1_dyt_fuse_enc_dec_4m_ft_bp_iter2b_sa_distill/default/ckpt_epoch_39.pth" --decoder --simplify --upsample