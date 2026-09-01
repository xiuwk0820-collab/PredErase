"""SAM predictor extension for official ReMOVE metric (CVPRW 2024)."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from segment_anything.modeling import Sam
from segment_anything.predictor import SamPredictor


class RemoveSamPredictor(SamPredictor):
    @torch.no_grad()
    def set_torch_image_return(
        self,
        transformed_image: torch.Tensor,
        original_image_size: tuple[int, ...],
    ) -> torch.Tensor:
        self.reset_image()
        self.original_size = original_image_size
        self.input_size = tuple(transformed_image.shape[-2:])
        input_image = self.model.preprocess(transformed_image)
        self.features = self.model.image_encoder(input_image)
        self.is_image_set = True
        return self.features

    @torch.no_grad()
    def get_aggregate_features(
        self,
        image: np.ndarray,
        masks: list[np.ndarray],
        image_format: str = "RGB",
    ) -> list[torch.Tensor]:
        assert image_format in ("RGB", "BGR")
        if image_format != self.model.image_format:
            image = image[..., ::-1]

        input_image = self.transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=self.device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

        features = self.set_torch_image_return(input_image_torch, image.shape[:2])
        emb: list[torch.Tensor] = []
        for mask in masks:
            mask_t = torch.as_tensor(mask, device=self.device).bool()
            active = features[mask_t.repeat((1, 256, 1, 1))].reshape((1, 256, -1))
            emb.append(active.mean(2))
        return emb
