# EdgeSAM-DyT
Say no to layernorms - combines [EdgeSAM](https://github.com/chongzhou96/EdgeSAM) with [Dynamic Tanh](https://github.com/jiachenzhu/DyT)

LayerNormalization presents a problem when quantizing due to its sensitivity to numerical accuracy. We replace layernorms in the image encoder and mask decoder with dynamic tanh (DyT) layers. This requires retraining via knowledge distillation of the image encoder (based on RepViT) and the mask decoder - here we present a 3 step distillation curriculum.

The goal is to create a fast and accurate general segmentation model for edge devices leveraging SAM. We encourage contributions to this effort.

This codebase adapts the code in the EdgeSAM repo.

## Updates
__2025-07-12__

We attempt to improve segmentation results with an **HQ** version of the model. Using the [datasets](https://huggingface.co/sam-hq-team/sam-hq-training/tree/main/data) listed in [sam-hq](https://github.com/SysCV/sam-hq) we distill the ViT-H HQ-SAM checkpoint.

Unlike HQ-SAM, since we use RepViT as the image encoder, we extract 3 intermediate convolutional layers, which are then fed to the [HQ MaskDecoder](./edge_sam/modeling/mask_decoder.py#L223). Layers needed to accomodate the intermediate features also use DyT layers.

Issues:
  - Training with amp float16 precision causes a NaN crash but float32 and bfloat16 work. We are not yet able to determine the reason (all previous steps trained with float16)
  - Training crashes consistently on epoch 41 (the config file specifies 48 epochs of training)

## Pretrained Models
We train against 5% of SA-1B and provide distillation checkpoints below:

STEP 1: [PyTorch](https://drive.google.com/file/d/14zMPCbdInahfwNS8rHIMdpY2m7szVwaw/view?usp=drive_link) 

STEP 2: [PyTorch](https://drive.google.com/file/d/1lsd2TsfYMgBN3NJGxGVs-HEu2DANaJQz/view?usp=drive_link) 

STEP 3 (Final Base Model): [PyTorch](https://drive.google.com/file/d/1YFBE939hOeraelSXm4lEYzoOLR-WQRtS/view?usp=drive_link) | [ONNX Encoder](https://drive.google.com/file/d/12jHKCPMymUqdQvh8BbSEcPcC3hYgxGS3/view?usp=drive_link) | [ONNX Decoder](https://drive.google.com/file/d/1SSovZSC95RcboqI7HQtmwXTFG3i_L4RV/view?usp=drive_link)

STEP 4 (OPTIONAL HQ Model): [PyTorch](https://drive.google.com/file/d/1RDYV3nQex9owmp28mxvg1d9yaQ-e1cJK/view?usp=drive_link) | [ONNX Encoder](https://drive.google.com/file/d/14twlBMSn-XH6hCP6GJHYLoDwAW3okV5O/view?usp=drive_link) | [ONNX Decoder](https://drive.google.com/file/d/1LxUCxf8NwLgsXc93KA2jB_BtmniOjhgV/view?usp=drive_link)

## Reproducibility
We provide the full script in `scripts/download_data_and_run_distillation.sh`

- Data:
  - links to training and validation are located in `training/sa_train_subset.txt` and `training/sa_val_subset.txt`
  - Note: The links may have changed. If links don't work check https://ai.meta.com/datasets/segment-anything-downloads/

- Image encoder:
  - We initialize from RepViT m0.9 (https://github.com/THU-MIG/RepViT)
  - We replace layernorms with DyT, and use `GELU(approximate='tanh')` during distillation (STEPS 1 and 2)

- Mask Decoder:
  - We replace all layernorms with DyT layers and use `GELU(approximate='tanh')`. This requires an extensive retraining of the mask decoder (STEP 3).

- Prompt Encoder:
  - We use the original prompt encoder from SAM.

## Installation
This should work with `torch>=2.0.0`
Currently tested with `torch==2.6.0` and `torchvision==0.21.0`
```
pip install torch==2.6.0
pip install torchvision==0.21.1
```
Then suggest building mmcv, mmengine, mmdet from source. See `scripts/install_mmcv.sh`

Then install the rest of packages in `requirements.txt`
```
pip install -r requirements.txt
```

## Web demo
```
cd ~/edgesam-dyt/web_demo/

PYTHONPATH=/home/ubuntu/edgesam-dyt/ python gradio_app.py --checkpoint "/home/ubuntu/edgesam-dyt/output/rep_vit_m1_dyt_fuse_enc_dec_4m_ft_bp_iter2b_sa_distill/default/ckpt_epoch_39.pth" --server-name=127.0.0.1 --port=7680
```
adding `--hq` will switch to edgesam-dyt-hq model
```
PYTHONPATH=/home/ubuntu/edgesam-dyt/ python gradio_app.py --checkpoint "/home/ubuntu/edgesam-dyt/output/rep_vit_m1_dyt_hq_fuse_enc_dec/default/ckpt_epoch_40.pth" --server-name=127.0.0.1 --port=7680 --hq
```

## Usage
See `notebooks/predictor_example.ipynb`
```
import sys
sys.path.append("..")
from edge_sam import sam_model_registry, SamPredictor
sam_checkpoint = "../output/rep_vit_m1_dyt_fuse_enc_dec_4m_ft_bp_iter2b_sa_distill/default/ckpt_epoch_39.pth"
model_type = "edge_sam_dyt"

device = "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)
```
If using the HQ version use `model_type = "edge_sam_dyt_hq"` and use the appropriate checkpoint.
See `notebooks/predictor_example.ipynb`

## ONNX
We update ONNX models to support both point and box prompts simulatenously.
Currently N points can be used but only 1 bounding box.
Additionally by adding `--upsample` flag to the conversion the mask results are interpolated from 256x256 to 1024x1024

See `notebooks/predictor_example_onnx.ipynb`
```
import sys
sys.path.append("..")
from edge_sam.onnx.predictor_onnx import SamPredictorONNX

predictor = SamPredictorONNX(
    encoder_path="../output/rep_vit_m1_dyt_fuse_enc_dec_4m_ft_bp_iter2b_sa_distill/default/ckpt_epoch_39_encoder.onnx",
    decoder_path="../output/rep_vit_m1_dyt_fuse_enc_dec_4m_ft_bp_iter2b_sa_distill/default/ckpt_epoch_39_decoder.onnx",
)

input_point = np.array([[500, 375], [1125, 625]])
input_label = np.array([1, 1])

masks, scores, logits = predictor.predict(
    point_coords=input_point[None], # specify N points of shape (1, N, 2)
    point_labels=input_label[None], # specify points labels of shape (1, N)
    point_valid=np.ones((1,1),bool), # whether to use point_coords inputs. must be bool
    boxes=np.zeros((1,1,4)), # specify bounding box of shape (1, 1, 4)
    boxes_valid=np.zeros((1,1),bool), # whether to use boxes input. must be bool
)

# mask shape (1, 4, H, W)
```

To support the HQ version we must expose the intermediate layers from the image encoder and provide new inputs for those layers in the mask decoder. 
For this `--hq` is appended to the onnx conversion command to indicate that new inputs/outputs are needed:
```
"interm_embeddings_0"
"interm_embeddings_1"
"interm_embeddings_2"
```
See the modification for the [encoder](./scripts/export_onnx_model.py#L75) and the [decoder](./scripts/export_onnx_model.py#L136)

## Conversion to TensorRT
```
trtexec --onnx=ckpt_epoch_39_encoder.onnx --fp16 --saveEngine=ckpt_epoch_39_encoder_fp16.trt --useCudaGraph

trtexec --onnx=ckpt_epoch_39_decoder.onnx --fp16 --saveEngine=ckpt_epoch_39_decoder_fp16.trt --useCudaGraph
```

NOTE: Usage code to follow soon...

## Conversion to TFLite for use on Android

NOTE: Demo will follow soon...

## Future Work
- [ ] Batchwise bounding box ONNX support
- [ ] Finetuning on smaller datasets
- [ ] Self prompting (Freeze Image Encoder/ Mask decoder - finetune prompt encoder for specific task)
- [x] Integration with [HQ-SAM](https://github.com/SysCV/sam-hq) for refined masks

## Citations
If this work helped you:
```
@software{edgesam_dyt,
  author = {Krasner, Alex},
  title = {EdgeSAM-DyT},
  url = {https://github.com/Krasner/edgesam-dyt},
  year = {2025}
}
```

Also cite all of these:

### EdgeSAM:
```bibtex
@article{zhou2023edgesam,
  title={EdgeSAM: Prompt-In-the-Loop Distillation for On-Device Deployment of SAM},
  author={Zhou, Chong and Li, Xiangtai and Loy, Chen Change and Dai, Bo},
  journal={arXiv preprint arXiv:2312.06660},
  year={2023}
}
```
### Dynamic Tanh:
```bibtex
@inproceedings{Zhu2025DyT,
  title={Transformers without Normalization},
  author={Zhu, Jiachen and Chen, Xinlei and He, Kaiming and LeCun, Yann and Liu, Zhuang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```
### RepViT:
```
@inproceedings{wang2024repvit,
  title={Repvit: Revisiting mobile cnn from vit perspective},
  author={Wang, Ao and Chen, Hui and Lin, Zijia and Han, Jungong and Ding, Guiguang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={15909--15920},
  year={2024}
}

@misc{wang2023repvitsam,
      title={RepViT-SAM: Towards Real-Time Segmenting Anything}, 
      author={Ao Wang and Hui Chen and Zijia Lin and Jungong Han and Guiguang Ding},
      year={2023},
      eprint={2312.05760},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
### SAM:
```
@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```
### HQ-SAM:
```
@inproceedings{sam_hq,
    title={Segment Anything in High Quality},
    author={Ke, Lei and Ye, Mingqiao and Danelljan, Martin and Liu, Yifan and Tai, Yu-Wing and Tang, Chi-Keung and Yu, Fisher},
    booktitle={NeurIPS},
    year={2023}
} 
```