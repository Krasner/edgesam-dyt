import torch
import torch.nn as nn
import torch.nn.functional as F
from ..modeling import Sam
from .amg import calculate_stability_score

from typing import Any, Optional, List

class SamCoreMLModel(nn.Module):
    """
    This model should not be called directly, but is used in CoreML export.
    """

    def __init__(
        self,
        model: Sam,
        use_stability_score: bool = False,
        upsample_masks: bool = False,
    ) -> None:
        super().__init__()
        self.mask_decoder = model.mask_decoder
        self.model = model
        self.img_size = model.image_encoder.img_size
        self.embed_dim = model.prompt_encoder.embed_dim
        self.use_stability_score = use_stability_score
        self.stability_score_offset = 1.0
        self.upsample = upsample_masks

    def _embed_points(self, point_coords: torch.Tensor, point_labels: torch.Tensor) -> torch.Tensor:
        point_coords = point_coords + 0.5
        point_coords = point_coords / self.img_size
        point_embedding = self.model.prompt_encoder.pe_layer._pe_encoding(point_coords)
        point_labels = point_labels.unsqueeze(-1).expand_as(point_embedding)

        point_embedding = point_embedding * (point_labels != -1)
        point_embedding = point_embedding + self.model.prompt_encoder.not_a_point_embed.weight * (
            point_labels == -1
        )

        for i in range(self.model.prompt_encoder.num_point_embeddings):
            point_embedding = point_embedding + self.model.prompt_encoder.point_embeddings[
                i
            ].weight * (point_labels == i)

        return point_embedding
    
    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Embeds box prompts."""
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)
        # corner_embedding = self.model.prompt_encoder.pe_layer.forward_with_coords(coords, self.img_size)
        coords = coords / self.img_size
        corner_embedding = self.model.prompt_encoder.pe_layer._pe_encoding(coords)

        corner_embedding[:, 0, :] += self.model.prompt_encoder.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.model.prompt_encoder.point_embeddings[3].weight
        return corner_embedding
    
    def _get_batch_size(
        self,
        points: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor],
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        if points is not None:
            return points.shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        else:
            return 1

    @torch.no_grad()
    def forward(
        self,
        image_embeddings: torch.Tensor,
        point_coords: torch.Tensor, # (B, N, 2)
        point_labels: torch.Tensor, # (B, N)
        boxes: torch.Tensor, # (B, M, 4)
        point_valid: torch.Tensor, # (B, 1)
        boxes_valid: torch.Tensor, # (B, 1)
    ):
        bs = self._get_batch_size(point_coords, boxes)
        # bs will likely be 1

        sparse_embeddings = torch.empty((bs, 0, self.embed_dim))
        
        point_embeddings = self._embed_points(point_coords, point_labels)
        box_embeddings = self._embed_boxes(boxes)

        point_valid = point_valid.squeeze(0)
        boxes_valid = boxes_valid.squeeze(0)
        # if point_coords is not None:
        # if point_valid:
        #     # breakpoint()
        #     sparse_embeddings = torch.cat([
        #         sparse_embeddings,
        #         point_embeddings,
        #     ], dim = 1)

        # if boxes is not None:
        # if boxes_valid:
        #     # breakpoint()
        #     sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)
        # breakpoint()

        # point_embeddings = point_embeddings[point_valid]
        # box_embeddings = box_embeddings[boxes_valid]
        # if point_embeddings.shape[0] == 0:
        #     point_embeddings = point_embeddings.reshape((bs, 0, self.embed_dim))
        # if box_embeddings.shape[0] == 0:
        #     box_embeddings = box_embeddings.reshape((bs, 0, self.embed_dim))
        point_valid = torch.tile(point_valid, (point_embeddings.shape[1], ))
        boxes_valid = torch.tile(boxes_valid, (box_embeddings.shape[1], ))

        point_embeddings = point_embeddings[:, point_valid]
        box_embeddings = box_embeddings[:,boxes_valid]

        """
        reshape box_embeddings if multiple boxes
        """
        # box_embeddings = box_embeddings.reshape((bs, -1, self.embed_dim))
        # breakpoint()
        sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings, box_embeddings], dim=1)
        
        dense_embedding = self.model.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)

        masks, scores = self.model.mask_decoder.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embedding,
        )

        if self.use_stability_score:
            scores = calculate_stability_score(
                masks, self.model.mask_threshold, self.stability_score_offset
            )

        if self.upsample:
            masks = masks.float()
            masks = F.interpolate(masks, (1024, 1024), mode='bilinear')

        return scores, masks

class SamCoreMLModelHQ(SamCoreMLModel):
    @torch.no_grad()
    def forward(
        self,
        image_embeddings: torch.Tensor,
        point_coords: torch.Tensor, # (B, N, 2)
        point_labels: torch.Tensor, # (B, N)
        boxes: torch.Tensor, # (B, M, 4)
        point_valid: torch.Tensor, # (B, 1)
        boxes_valid: torch.Tensor, # (B, 1)
        # interm_embeddings: List[torch.Tensor],
        interm_embeddings_0: torch.Tensor,
        interm_embeddings_1: torch.Tensor,
        interm_embeddings_2: torch.Tensor,
    ):
        bs = self._get_batch_size(point_coords, boxes)
        # bs will likely be 1

        sparse_embeddings = torch.empty((bs, 0, self.embed_dim))
        
        point_embeddings = self._embed_points(point_coords, point_labels)
        box_embeddings = self._embed_boxes(boxes)

        point_valid = point_valid.squeeze(0)
        boxes_valid = boxes_valid.squeeze(0)
        
        point_valid = torch.tile(point_valid, (point_embeddings.shape[1], ))
        boxes_valid = torch.tile(boxes_valid, (box_embeddings.shape[1], ))

        point_embeddings = point_embeddings[:, point_valid]
        box_embeddings = box_embeddings[:,boxes_valid]

        interm_embeddings = [
            interm_embeddings_0,
            interm_embeddings_1,
            interm_embeddings_2,
        ]
        """
        reshape box_embeddings if multiple boxes
        """
        # box_embeddings = box_embeddings.reshape((bs, -1, self.embed_dim))
        # breakpoint()
        sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings, box_embeddings], dim=1)
        
        dense_embedding = self.model.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)

        hq_features = (
            self.model.mask_decoder.embedding_encoder(image_embeddings) + 
            sum([self.model.mask_decoder.compress_vit_feat[i](int_emb) for i, int_emb in enumerate(interm_embeddings)])
        )

        masks, scores = self.model.mask_decoder.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embedding,
            hq_features = hq_features,
        )

        if self.use_stability_score:
            scores = calculate_stability_score(
                masks, self.model.mask_threshold, self.stability_score_offset
            )

        if self.upsample:
            masks = masks.float()
            masks = F.interpolate(masks, (1024, 1024), mode='bilinear')

        return scores, masks