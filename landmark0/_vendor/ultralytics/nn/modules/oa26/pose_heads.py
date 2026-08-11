# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Experimental OA26 pose heads.

These heads keep the standard YOLO26 pose regression output intact and add
training-only auxiliary landmark0 supervision branches.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import Pose26
from ultralytics.utils.oa26.simcc import decode_simcc_logits


class OA26HeatmapPose(Pose26):
    """YOLO26 pose head with an auxiliary image-level heatmap branch."""

    def __init__(
        self,
        nc: int = 80,
        kpt_shape: tuple = (129, 3),
        heatmap_channels: int = 0,
        heatmap_temperature: float = 1.0,
        reg_max: int = 1,
        end2end: bool = False,
        ch: tuple = (),
    ):
        """Initialize the head.

        Args are ordered to match `parse_model()`, which appends
        `[reg_max, end2end, ch]` after YAML-specified arguments.
        """
        super().__init__(nc, kpt_shape, reg_max, end2end, ch)
        self.heatmap_channels = int(heatmap_channels or kpt_shape[0])
        self.heatmap_temperature = float(heatmap_temperature)

        c = max(ch[0] // 2, self.heatmap_channels)
        self.hm_head = nn.Sequential(
            Conv(ch[0], c, 3),
            Conv(c, c, 3),
            nn.Conv2d(c, self.heatmap_channels, 1),
        )
        if end2end:
            self.one2one_hm_head = copy.deepcopy(self.hm_head)

    @property
    def one2many(self):
        """Return one-to-many head modules, including the auxiliary heatmap branch."""
        heads = super().one2many
        heads["heatmap_head"] = self.hm_head
        return heads

    @property
    def one2one(self):
        """Return one-to-one head modules, including the auxiliary heatmap branch."""
        heads = super().one2one
        heads["heatmap_head"] = self.one2one_hm_head
        return heads

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: torch.nn.Module,
        cls_head: torch.nn.Module,
        pose_head: torch.nn.Module,
        kpts_head: torch.nn.Module,
        kpts_sigma_head: torch.nn.Module,
        heatmap_head: torch.nn.Module | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return standard pose predictions plus auxiliary heatmaps."""
        preds = super().forward_head(x, box_head, cls_head, pose_head, kpts_head, kpts_sigma_head)
        if heatmap_head is not None:
            heatmaps = heatmap_head(x[0])  # BCHW, B x K x H/4 x W/4 for the OA26 P2 head.
            preds["heatmaps"] = heatmaps
            preds["hm_kpts"] = self.soft_argmax_2d(heatmaps)
        return preds

    def soft_argmax_2d(self, heatmaps: torch.Tensor) -> torch.Tensor:
        """Decode heatmaps to feature-map coordinates using soft-argmax."""
        b, k, h, w = heatmaps.shape
        logits = heatmaps.view(b, k, -1) / max(self.heatmap_temperature, 1e-6)
        prob = logits.softmax(dim=-1)
        dtype, device = heatmaps.dtype, heatmaps.device
        ys = torch.arange(h, device=device, dtype=dtype)
        xs = torch.arange(w, device=device, dtype=dtype)
        y_grid, x_grid = torch.meshgrid(ys, xs, indexing="ij")
        x = (prob * x_grid.reshape(1, 1, -1)).sum(dim=-1)
        y = (prob * y_grid.reshape(1, 1, -1)).sum(dim=-1)
        return torch.stack((x, y), dim=-1)

    def fuse(self) -> None:
        """Remove auxiliary one-to-many branches for optimized inference."""
        super().fuse()
        self.hm_head = None


class OA26SimCCPose(Pose26):
    """YOLO26 pose head with an auxiliary SimCC x/y classification branch."""

    def __init__(
        self,
        nc: int = 80,
        kpt_shape: tuple = (129, 3),
        simcc_imgsz: int | tuple[int, int] = 896,
        simcc_split_ratio: float = 2.0,
        reg_max: int = 1,
        end2end: bool = False,
        ch: tuple = (),
    ):
        """Initialize the head with standard Pose26 outputs and SimCC logits."""
        super().__init__(nc, kpt_shape, reg_max, end2end, ch)
        self.simcc_imgsz = self._pair(simcc_imgsz)
        self.simcc_split_ratio = float(simcc_split_ratio)
        self.simcc_bins = (
            max(int(round(self.simcc_imgsz[0] * self.simcc_split_ratio)), 1),
            max(int(round(self.simcc_imgsz[1] * self.simcc_split_ratio)), 1),
        )  # (height_bins, width_bins)

        c = max(ch[0] // 2, kpt_shape[0])
        self.simcc_head = nn.Sequential(
            Conv(ch[0], c, 3),
            Conv(c, c, 3),
            nn.Conv2d(c, kpt_shape[0], 1),
        )
        if end2end:
            self.one2one_simcc_head = copy.deepcopy(self.simcc_head)

    @staticmethod
    def _pair(value: int | tuple[int, int] | list[int]) -> tuple[int, int]:
        """Normalize image size config to `(height, width)`."""
        if isinstance(value, (tuple, list)):
            return int(value[0]), int(value[1])
        return int(value), int(value)

    @property
    def one2many(self):
        """Return one-to-many head modules, including the auxiliary SimCC branch."""
        heads = super().one2many
        heads["simcc_head"] = self.simcc_head
        return heads

    @property
    def one2one(self):
        """Return one-to-one head modules, including the auxiliary SimCC branch."""
        heads = super().one2one
        heads["simcc_head"] = self.one2one_simcc_head
        return heads

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: torch.nn.Module,
        cls_head: torch.nn.Module,
        pose_head: torch.nn.Module,
        kpts_head: torch.nn.Module,
        kpts_sigma_head: torch.nn.Module,
        simcc_head: torch.nn.Module | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return standard pose predictions plus auxiliary SimCC logits."""
        preds = super().forward_head(x, box_head, cls_head, pose_head, kpts_head, kpts_sigma_head)
        if simcc_head is not None:
            simcc_map = simcc_head(x[0])  # B x K x H x W
            y_bins, x_bins = self.simcc_bins
            x_logits = F.interpolate(simcc_map.mean(dim=2), size=x_bins, mode="linear", align_corners=False)
            y_logits = F.interpolate(simcc_map.mean(dim=3), size=y_bins, mode="linear", align_corners=False)
            preds["simcc_x"] = x_logits
            preds["simcc_y"] = y_logits
            preds["simcc_kpts"] = self.decode_simcc(x_logits, y_logits)
        return preds

    def decode_simcc(self, x_logits: torch.Tensor, y_logits: torch.Tensor) -> torch.Tensor:
        """Decode SimCC logits to `[B, K, 3]` image-space keypoints for debugging/loss support."""
        return decode_simcc_logits(x_logits, y_logits, self.simcc_split_ratio)

    def fuse(self) -> None:
        """Remove auxiliary one-to-many branches for optimized inference."""
        super().fuse()
        self.simcc_head = None
