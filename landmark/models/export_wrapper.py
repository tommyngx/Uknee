from __future__ import annotations

from torch import nn


class LandmarkExportWrapper(nn.Module):
    """Tensor-only export interface for ONNX/TorchScript deployment."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image):
        outputs = self.model(
            image,
            return_heatmaps=False,
            return_features=False,
        )
        segmentation = outputs.get("segmentation_logits")
        if segmentation is None:
            # Baselines do not predict segmentation; preserve a tensor-only tuple.
            segmentation = image[:, :0]
        return (
            segmentation,
            outputs["final_landmarks"],
            outputs["landmark_confidence"],
        )
