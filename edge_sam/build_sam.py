# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from functools import partial

import edge_sam.modeling as modeling
from edge_sam.modeling import (
    ImageEncoderViT, 
    MaskDecoder, 
    PromptEncoder, 
    Sam, 
    TwoWayTransformer, 
    RepViT,
    SamBatch, 
    PromptEncoderBatch, 
    MaskDecoderBatch,
    MaskDecoderHQ,
)
from edge_sam.config import _C, _update_config_from_file
from yacs.config import CfgNode as CN


prompt_embed_dim = 256
image_size = 1024
vit_patch_size = 16
image_embedding_size = image_size // vit_patch_size


def build_sam_vit_h(checkpoint=None, **kwargs):
    image_encoder = _build_sam_encoder(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31]
    )
    if kwargs.pop('encoder_only', False):
        return image_encoder
    return _build_sam(image_encoder, checkpoint, **kwargs)


def build_sam_vit_l(checkpoint=None, **kwargs):
    image_encoder = _build_sam_encoder(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23]
    )
    if kwargs.pop('encoder_only', False):
        return image_encoder
    return _build_sam(image_encoder, checkpoint, **kwargs)


def build_sam_vit_b(checkpoint=None, **kwargs):
    image_encoder = _build_sam_encoder(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11]
    )
    if kwargs.pop('encoder_only', False):
        return image_encoder
    return _build_sam(image_encoder, checkpoint, **kwargs)


def build_edge_sam(checkpoint=None, upsample_mode="bicubic"):
    image_encoder = RepViT(
        arch="m1",
        img_size=image_size,
        upsample_mode=upsample_mode,
        fuse=True
    )
    return _build_sam(image_encoder, checkpoint)

def build_edge_sam_dyt(checkpoint=None, upsample_mode="bicubic"):
    image_encoder = RepViT(
        arch="m1",
        img_size=image_size,
        upsample_mode=upsample_mode,
        fuse=True,
        dyt=True,
        gelu_approx='tanh',
    )
    return _build_sam(image_encoder, checkpoint, dyt=True, gelu_approx='tanh')

def build_edge_sam_dyt_hq(checkpoint=None, upsample_mode="bicubic"):
    image_encoder = RepViT(
        arch="m1",
        img_size=image_size,
        upsample_mode=upsample_mode,
        fuse=True,
        dyt=True,
        gelu_approx='tanh',
        return_low_feats=True,
        feature_names = ['stem','stage0','stage1']
    )
    return _build_sam(image_encoder, checkpoint, dyt=True, gelu_approx='tanh', enable_hq=True)


def build_edge_sam_dyt_frozen_mask(checkpoint=None, upsample_mode="bicubic"):
    image_encoder = RepViT(
        arch="m1",
        img_size=image_size,
        upsample_mode=upsample_mode,
        fuse=True,
        dyt=True,
        gelu_approx='tanh',
    )
    return _build_sam(image_encoder, checkpoint, dyt=False, gelu_approx='none')

sam_model_registry = {
    "default": build_edge_sam,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
    "edge_sam": build_edge_sam,
    "edge_sam_dyt": build_edge_sam_dyt,
    "edge_sam_dyt_hq": build_edge_sam_dyt_hq,
    "edge_sam_dyt_frozen_mask": build_edge_sam_dyt_frozen_mask,
}
build_sam = build_edge_sam


def _build_sam_encoder(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
):
    image_encoder = ImageEncoderViT(
        depth=encoder_depth,
        embed_dim=encoder_embed_dim,
        img_size=image_size,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=encoder_num_heads,
        patch_size=vit_patch_size,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=encoder_global_attn_indexes,
        window_size=14,
        out_chans=prompt_embed_dim,
    )
    return image_encoder


def _build_sam(image_encoder, checkpoint, enable_batch=False, enable_distill=False, lora=False, rpn_head=None, dyt=False, gelu_approx='none', enable_hq=False):
    sam_model = SamBatch if enable_batch else Sam
    prompt_encoder = PromptEncoderBatch if enable_batch else PromptEncoder
    if enable_hq:
        mask_decoder = MaskDecoderHQ
    elif enable_batch:
        mask_decoder = MaskDecoderBatch
    else:
        mask_decoder = MaskDecoder

    sam_args = dict(
        image_encoder=image_encoder,
        prompt_encoder=prompt_encoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=mask_decoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
                lora=lora,
                dyt=dyt,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            yield_kd_targets=enable_distill,
            dyt=dyt,
            gelu_approx=gelu_approx,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )

    if rpn_head != 'none':
        sam_args['rpn_head'] = rpn_head

    sam = sam_model(**sam_args)

    if not enable_distill:
        sam.eval()
        if checkpoint is not None:
            with open(checkpoint, "rb") as f:
                state_dict = torch.load(f, weights_only=False)
                if 'model' in state_dict:
                    state_dict = state_dict['model']
            print(sam.load_state_dict(state_dict, strict=False))
    return sam


# build sam model from the yaml config file
def build_sam_from_config(
        cfg_file,
        checkpoint=None,
        enable_distill=False,
        enable_batch=False,
        use_original_mask_decoder=False,
        **kwargs):
    if isinstance(cfg_file, CN):
        config = cfg_file
    else:
        config = _C.clone()
        _update_config_from_file(config, cfg_file)
    # breakpoint()
    model_type = config.MODEL.TYPE
    encoder_only = config.DISTILL.ENCODER_ONLY
    lora = config.DISTILL.LORA
    fuse = config.DISTILL.FUSE
    rpn_head = config.DISTILL.RPN_HEAD

    if not use_original_mask_decoder:
        dyt = config.DISTILL.MASK_DYT
        gelu_approx = config.DISTILL.GELU_APPROX
        enable_hq = config.MODEL.MASK_HQ
    else:
        dyt = False
        gelu_approx = 'none'
        enable_hq = False

    if model_type in ['vit_h', 'vit_l', 'vit_b']:
        return sam_model_registry[model_type](
            checkpoint,
            enable_distill=enable_distill,
            enable_batch=enable_batch,
            encoder_only=encoder_only,
            lora=lora
        )

    kwargs['upsample_mode'] = config.DISTILL.UPSAMPLE_MODE
    if config.MODEL.MASK_HQ:
        kwargs['return_low_feats'] = True

    image_encoder = getattr(modeling, model_type)(fuse=fuse, **kwargs)

    if encoder_only:
        return image_encoder

    sam = _build_sam(image_encoder, checkpoint, enable_batch, enable_distill, lora, rpn_head, dyt, gelu_approx, enable_hq)
    return sam
