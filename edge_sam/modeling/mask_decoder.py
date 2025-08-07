# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type, Optional

from .common import LayerNorm2d, DynamicTanh


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        yield_kd_targets=False,
        dyt=False,
        gelu_approx='none',
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
          dyt (bool): use DyT instead of LayerNorm
          gelu_approx (str): 'none' or 'tanh'
        """
        super().__init__()
        self.yield_kd_targets = yield_kd_targets
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        if dyt:
            self.output_upscaling = nn.Sequential(
                nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
                DynamicTanh(transformer_dim // 4, False),
                activation(gelu_approx),
                nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
                activation(gelu_approx),
            )
        else:
            self.output_upscaling = nn.Sequential(
                nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
                LayerNorm2d(transformer_dim // 4),
                activation(gelu_approx),
                nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
                activation(gelu_approx),
            )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        num_multimask_outputs: int,
        num_prompts=None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """

        if self.yield_kd_targets:
            kd_targets = dict()
        else:
            kd_targets = None

        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            num_prompts=num_prompts,
            kd_targets=kd_targets
        )

        # Select the correct mask or masks for output
        if num_multimask_outputs == 4:
            mask_slice = slice(0, None)
        elif num_multimask_outputs == 3:
            mask_slice = slice(1, None)
        elif num_multimask_outputs == 1:
            mask_slice = slice(0, 1)
        else:
            raise ValueError

        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]
        if kd_targets is not None:
            kd_targets['query'] = kd_targets['query'][:, mask_slice]
        if self.yield_kd_targets:
            return masks, iou_pred, kd_targets
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        num_prompts=None,
        kd_targets=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        if tokens.shape[0] > 1:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            src = image_embeddings
        src = src + dense_prompt_embeddings
        if tokens.shape[0] > 1:
            pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        else:
            pos_src = image_pe
            
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens, kd_targets)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)

        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        if self.yield_kd_targets:
            kd_targets['query'] = hyper_in
            kd_targets['feat'] = upscaled_embedding
        return masks, iou_pred


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x

class MaskDecoderHQ(MaskDecoder):
    def __init__(
        self,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        yield_kd_targets=False,
        dyt=False,
        gelu_approx='none',
        vit_dim=[48,48,96],
    ) -> None:
        """
        Low level feature will be of shape: 
        stem - (B, 48, 256, 256)
        stage0 - (B, 48, 256, 256)
        stage1 - (B, 96, 128, 128)
        Unlike HQ-SAM which expects (B, 1280, 64, 64) which then requires Conv2dTranspose upsampling
        """
        super().__init__(
            transformer_dim=transformer_dim,
            transformer=transformer,
            num_multimask_outputs=num_multimask_outputs,
            activation=activation,
            iou_head_depth=iou_head_depth,
            iou_head_hidden_dim=iou_head_hidden_dim,
            yield_kd_targets=yield_kd_targets,
            dyt=dyt,
            gelu_approx=gelu_approx,
        )

        self.hf_token = nn.Embedding(1, transformer_dim)
        self.hf_mlp = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
        self.num_mask_tokens = self.num_mask_tokens + 1

        if dyt:
            self.compress_vit_feat = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(vit_dim[i], transformer_dim, 3, 1, 1) if i < 2 else nn.ConvTranspose2d(vit_dim[i], transformer_dim, kernel_size=2, stride=2),
                    DynamicTanh(transformer_dim, False),
                    activation(gelu_approx), 
                    nn.Conv2d(transformer_dim, transformer_dim // 8, 3, 1, 1),
                    # activation(gelu_approx),
                ) 
                for i in range(len(vit_dim))
            ])
            
            self.embedding_encoder = nn.Sequential(
                                            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
                                            DynamicTanh(transformer_dim // 4, False),
                                            activation(gelu_approx),
                                            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
                                            # activation(gelu_approx),
                                        )

            self.embedding_maskfeature = nn.Sequential(
                                            nn.Conv2d(transformer_dim // 8, transformer_dim // 4, 3, 1, 1), 
                                            DynamicTanh(transformer_dim // 4, False),
                                            activation(gelu_approx),
                                            nn.Conv2d(transformer_dim // 4, transformer_dim // 8, 3, 1, 1),
                                            # activation(gelu_approx),
                                        )

        else:
            self.compress_vit_feat = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(vit_dim[i], transformer_dim, 3, 1, 1) if i < 2 else nn.ConvTranspose2d(vit_dim[i], transformer_dim, kernel_size=2, stride=2),
                    LayerNorm2d(transformer_dim),
                    activation(gelu_approx), 
                    nn.Conv2d(transformer_dim, transformer_dim // 8, 3, 1, 1)
                )
                for i in range(len(vit_dim))
            ])
            
            self.embedding_encoder = nn.Sequential(
                                            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
                                            LayerNorm2d(transformer_dim // 4),
                                            activation(gelu_approx),
                                            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
                                        )

            self.embedding_maskfeature = nn.Sequential(
                                            nn.Conv2d(transformer_dim // 8, transformer_dim // 4, 3, 1, 1), 
                                            LayerNorm2d(transformer_dim // 4),
                                            activation(gelu_approx),
                                            nn.Conv2d(transformer_dim // 4, transformer_dim // 8, 3, 1, 1))

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        num_multimask_outputs: int,
        # multimask_output: bool,
        hq_token_only: bool = True,
        interm_embeddings: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the ViT image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted hq masks
        """
        
        # this will be (b, 64, 64, 1280) permuted to (b, 1280, 64, 64)
        # vit_features = interm_embeddings[0].permute(0, 3, 1, 2) # early-layer ViT feature, after 1st global attention block in ViT
        # print(vit_features.shape)

        hq_features = self.embedding_encoder(image_embeddings) + sum([self.compress_vit_feat[i](int_emb) for i, int_emb in enumerate(interm_embeddings)])
        # print(f"{hq_features.shape=}")
        # batch_len = len(image_embeddings)
        # masks = []
        # iou_preds = []
        # for i_batch in range(batch_len):
        #     mask, iou_pred = self.predict_masks(
        #         image_embeddings=image_embeddings[i_batch].unsqueeze(0),
        #         image_pe=image_pe[i_batch],
        #         sparse_prompt_embeddings=sparse_prompt_embeddings[i_batch],
        #         dense_prompt_embeddings=dense_prompt_embeddings[i_batch],
        #         hq_feature = hq_features[i_batch].unsqueeze(0)
        #     )
        #     masks.append(mask)
        #     iou_preds.append(iou_pred)
        # masks = torch.cat(masks,0)
        # iou_preds = torch.cat(iou_preds,0)
        masks, iou_preds = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            hq_features = hq_features,
        )

        # Select the correct mask or masks for output
        """
        if num_multimask_outputs == 4:
            mask_slice = slice(0, None)
        elif num_multimask_outputs == 3:
            mask_slice = slice(1, None)
        elif num_multimask_outputs == 1:
            mask_slice = slice(0, 1)
        else:
            raise ValueError

        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]
        if kd_targets is not None:
            kd_targets['query'] = kd_targets['query'][:, mask_slice]
        if self.yield_kd_targets:
            return masks, iou_pred, kd_targets
        return masks, iou_pred
        """
        # Select the correct mask or masks for output
        # if multimask_output:
        #     # mask with highest score
        #     mask_slice = slice(1,self.num_mask_tokens-1)
        #     iou_preds = iou_preds[:, mask_slice]
        #     iou_preds, max_iou_idx = torch.max(iou_preds,dim=1)
        #     iou_preds = iou_preds.unsqueeze(1)
        #     masks_multi = masks[:, mask_slice, :, :]
        #     masks_sam = masks_multi[torch.arange(masks_multi.size(0)),max_iou_idx].unsqueeze(1)
        # else:
        # single mask output, default
        mask_slice = slice(0, 1)
        masks_sam = masks[:,mask_slice]
        iou_preds = iou_preds[:, mask_slice]

        masks_hq = masks[:,slice(self.num_mask_tokens-1, self.num_mask_tokens)]
        
        # if hq_token_only:
        #     return masks_hq, iou_preds
        # else:
        #     # raise NotImplementedError
        return masks_hq, masks_sam, iou_preds
    """
    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        hq_feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Predicts masks. See 'forward' for more details.

        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight, self.hf_token.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0) 
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)

        upscaled_embedding_sam = self.output_upscaling(src)
        upscaled_embedding_ours = self.embedding_maskfeature(upscaled_embedding_sam) + hq_feature
        
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            if i < 4:
                hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
            else:
                hyper_in_list.append(self.hf_mlp(mask_tokens_out[:, i, :]))

        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding_sam.shape

        masks_sam = (hyper_in[:,:4] @ upscaled_embedding_sam.view(b, c, h * w)).view(b, -1, h, w)
        masks_ours = (hyper_in[:,4:] @ upscaled_embedding_ours.view(b, c, h * w)).view(b, -1, h, w)
        # if torch.any(torch.isnan(masks_ours)):
        #     breakpoint()

        masks = torch.cat([masks_sam,masks_ours],dim=1)
        
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred
    """
    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        hq_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight, self.hf_token.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)

        upscaled_embedding_sam = self.output_upscaling(src)
        upscaled_embedding_hq = self.embedding_maskfeature(upscaled_embedding_sam) + hq_features.repeat(b,1,1,1)

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            if i < self.num_mask_tokens - 1:
                hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
            else:
                hyper_in_list.append(self.hf_mlp(mask_tokens_out[:, i, :]))

        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding_sam.shape

        masks_sam = (hyper_in[:,:self.num_mask_tokens-1] @ upscaled_embedding_sam.view(b, c, h * w)).view(b, -1, h, w)
        masks_sam_hq = (hyper_in[:,self.num_mask_tokens-1:] @ upscaled_embedding_hq.view(b, c, h * w)).view(b, -1, h, w)
        masks = torch.cat([masks_sam,masks_sam_hq],dim=1)
        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred
