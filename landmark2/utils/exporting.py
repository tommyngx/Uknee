"""Stable deployment output wrapper shared by all five public models."""

from __future__ import annotations

import torch
from torch import nn

from landmark2.data.schema import REGION_KEYPOINT_COUNTS, REGION_NAMES


class _HeatmapHeadMetadata(nn.Module):
    kpt_shape = [51, 3]

    def forward(self, x):
        return x


class KneePoseExportWrapper(nn.Module):
    """Return ``detections``, ``num_detections`` and canonical 129 keypoints.

    ``detections`` has fixed shape ``[B, 4, 159]`` (xyxy, score, class,
    51x3 keypoints). Missing classes are zero-filled. ``canonical`` has fixed
    shape ``[B, 129, 3]``.
    """

    def __init__(self, core: nn.Module, family: str, confidence: float = 0.25):
        super().__init__()
        self.core = core
        self.family = family
        self.confidence = float(confidence)
        self.yaml = getattr(core, "yaml", {"channels": 3})
        self.yaml_file = getattr(core, "yaml_file", self.yaml.get("yaml_file", "uknee-pose.yaml"))
        self.names = getattr(core, "names", dict(enumerate(REGION_NAMES)))
        self.task = "pose"
        self.stride = getattr(core, "stride", torch.tensor([32.0]))
        core_args = getattr(core, "args", {})
        self.args = dict(core_args) if isinstance(core_args, dict) else vars(core_args)
        self.pt_path = getattr(core, "pt_path", None)
        self.end2end = True
        self.uknee_export_contract = True
        # Exporter metadata expects model[-1].kpt_shape.
        self.model = core.model if family == "yolo" else nn.ModuleList([_HeatmapHeadMetadata()])

    def forward(self, images: torch.Tensor):
        raw = self.core(images)
        raw = raw[0] if isinstance(raw, (tuple, list)) else raw
        if self.family == "heatmap":
            detections, present = self._heatmap_detections(raw, images.shape[-2], images.shape[-1])
        else:
            detections, present = self._select_yolo_classes(raw)
        canonical = self._canonical(detections, present)
        return detections, present.sum(dim=1).to(torch.int64), canonical

    def _select_yolo_classes(self, predictions: torch.Tensor):
        # End-to-end Pose26 export is B x max_det x (6 + 51*3).
        classes = predictions[..., 5]
        confidence = predictions[..., 4]
        rows, present = [], []
        batch = torch.arange(predictions.shape[0], device=predictions.device)
        for class_id in range(4):
            matches = classes == float(class_id)
            masked = torch.where(matches, confidence, confidence.new_full((), -1.0))
            score, index = masked.max(dim=1)
            selected = predictions[batch, index]
            valid = matches.any(dim=1) & (score >= self.confidence)
            selected = torch.where(valid[:, None], selected, torch.zeros_like(selected))
            rows.append(selected)
            present.append(valid)
        return torch.stack(rows, dim=1), torch.stack(present, dim=1)

    def _heatmap_detections(self, canonical: torch.Tensor, height: int, width: int):
        rows, present = [], []
        offset = 0
        for class_id, count in enumerate(REGION_KEYPOINT_COUNTS):
            local = canonical.new_zeros((canonical.shape[0], 51, 3))
            local[:, :count] = canonical[:, offset : offset + count]
            local[:, :count, 0] *= width
            local[:, :count, 1] *= height
            points = local[:, :count, :2]
            boxes = torch.cat((points.amin(dim=1), points.amax(dim=1)), dim=1)
            score = local[:, :count, 2].mean(dim=1)
            class_column = score.new_full((score.shape[0], 1), float(class_id))
            row = torch.cat((boxes, score[:, None], class_column, local.flatten(1)), dim=1)
            valid = score >= self.confidence
            rows.append(torch.where(valid[:, None], row, torch.zeros_like(row)))
            present.append(valid)
            offset += count
        return torch.stack(rows, dim=1), torch.stack(present, dim=1)

    @staticmethod
    def _canonical(detections: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        keypoints = detections[..., 6:].view(detections.shape[0], 4, 51, 3)
        chunks = []
        for class_id, count in enumerate(REGION_KEYPOINT_COUNTS):
            chunk = keypoints[:, class_id, :count]
            chunks.append(torch.where(present[:, class_id, None, None], chunk, torch.zeros_like(chunk)))
        return torch.cat(chunks, dim=1)

    def fuse(self, verbose: bool = False):
        fused = self.core.fuse(verbose=verbose) if self.family == "yolo" else self.core.fuse(verbose=verbose)
        if fused is not None:
            self.core = fused
        return self

    def info(self, detailed: bool = False, verbose: bool = True, imgsz: int = 640):
        return self.core.info(detailed=detailed, verbose=verbose, imgsz=imgsz)


__all__ = ["KneePoseExportWrapper"]
