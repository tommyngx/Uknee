"""Dependency-free RTMO adaptation for four knee regions and 129 landmarks.

The implementation retains the defining RTMO components from MMPose's
``RTMOHead``: multi-level one-stage candidates, split classification/pose
branches, grouped pose convolutions, and Dynamic Coordinate Classification
(DCC) with sine bin encoding and a Gated Attention Unit (GAU). Candidate
selection is specialized for Uknee's exactly four anatomical instances, which
avoids human-pose NMS/SimOTA assumptions that do not fit this dataset.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from landmark.data.schema import NUM_LANDMARKS, REGION_KEYPOINT_COUNTS


class ConvNormAct(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int, kernel: int = 3, stride: int = 1, groups: int = 1):
        padding = kernel // 2
        super().__init__(
            nn.Conv2d(input_channels, output_channels, kernel, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(output_channels, momentum=0.03, eps=0.001),
            nn.SiLU(inplace=True),
        )


class CSPBlock(nn.Module):
    def __init__(self, channels: int, repeats: int = 2):
        super().__init__()
        hidden = channels // 2
        self.left = ConvNormAct(channels, hidden, 1)
        self.right = ConvNormAct(channels, hidden, 1)
        self.blocks = nn.Sequential(
            *(nn.Sequential(ConvNormAct(hidden, hidden, 3), ConvNormAct(hidden, hidden, 3)) for _ in range(repeats))
        )
        self.merge = ConvNormAct(channels, channels, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        value = self.left(inputs)
        for block in self.blocks:
            value = value + block(value)
        return self.merge(torch.cat((value, self.right(inputs)), dim=1))


class RTMOBackbone(nn.Module):
    """CSPDarknet-style backbone producing stride 8/16/32 feature maps."""

    def __init__(self, input_channels: int = 3, width: int = 48, depth: int = 2):
        super().__init__()
        channels = (width, width * 2, width * 4, width * 8, width * 16)
        self.stem = ConvNormAct(input_channels, channels[0], 3, 2)
        self.stages = nn.ModuleList()
        previous = channels[0]
        for output in channels[1:]:
            self.stages.append(nn.Sequential(ConvNormAct(previous, output, 3, 2), CSPBlock(output, depth)))
            previous = output
        self.out_channels = channels[2:]

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        value = self.stem(images)
        outputs = []
        for index, stage in enumerate(self.stages):
            value = stage(value)
            if index >= 1:
                outputs.append(value)
        return tuple(outputs)


class HybridEncoder(nn.Module):
    """Lightweight RTMO hybrid encoder with deep attention and bidirectional FPN."""

    def __init__(self, in_channels: Sequence[int], channels: int = 256, attention_heads: int = 8):
        super().__init__()
        self.projections = nn.ModuleList(ConvNormAct(value, channels, 1) for value in in_channels)
        self.deep_norm = nn.LayerNorm(channels)
        self.deep_attention = nn.MultiheadAttention(channels, attention_heads, batch_first=True)
        self.top_down = CSPBlock(channels, 1)
        self.downsample = ConvNormAct(channels, channels, 3, 2)
        self.bottom_up = CSPBlock(channels, 1)

    def forward(self, features: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        _, middle, deep = [projection(value) for projection, value in zip(self.projections, features)]
        batch, channels, height, width = deep.shape
        tokens = deep.flatten(2).transpose(1, 2)
        normalized = self.deep_norm(tokens)
        tokens = tokens + self.deep_attention(normalized, normalized, normalized, need_weights=False)[0]
        deep = tokens.transpose(1, 2).reshape(batch, channels, height, width)
        stride16 = self.top_down(middle + F.interpolate(deep, middle.shape[-2:], mode="nearest"))
        stride32 = self.bottom_up(deep + self.downsample(stride16))
        return stride16, stride32


class ScaleNorm(nn.Module):
    """ScaleNorm used by MMPose's GAU implementation."""

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.scale = channels**-0.5
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        norm = torch.linalg.vector_norm(inputs, dim=-1, keepdim=True) * self.scale
        return inputs / norm.clamp_min(self.eps) * self.gain


class GAUEncoder(nn.Module):
    """Gated Attention Unit following the RTMO DCC token mixer."""

    def __init__(self, channels: int, attention_channels: int = 128, expansion_factor: int = 2):
        super().__init__()
        expanded = channels * expansion_factor
        self.norm = ScaleNorm(channels)
        self.expanded = expanded
        self.attention_channels = attention_channels
        self.uv = nn.Linear(channels, expanded * 2 + attention_channels, bias=False)
        self.gamma = nn.Parameter(torch.rand(2, attention_channels))
        self.beta = nn.Parameter(torch.rand(2, attention_channels))
        self.output = nn.Linear(expanded, channels, bias=False)
        self.residual_scale = nn.Parameter(torch.ones(channels))

    def forward(self, inputs: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(inputs)
        u, value, base = torch.split(
            F.silu(self.uv(normalized)),
            [self.expanded, self.expanded, self.attention_channels],
            dim=-1,
        )
        query = base * self.gamma[0] + self.beta[0] + position
        key = base * self.gamma[1] + self.beta[1] + position
        kernel = F.relu(torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.attention_channels)).square()
        mixed = u * torch.matmul(kernel, value)
        return inputs * self.residual_scale + self.output(mixed)


class SinePositionEncoding(nn.Module):
    def __init__(self, channels: int = 128, temperature: float = 300.0):
        super().__init__()
        if channels % 2:
            raise ValueError("Sine encoding channels must be even")
        frequencies = temperature ** (-torch.arange(channels // 2, dtype=torch.float32) / (channels // 2))
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        phase = positions.unsqueeze(-1) * self.frequencies
        return torch.cat((phase.cos(), phase.sin()), dim=-1)


class DynamicCoordinateClassifier(nn.Module):
    """RTMO DCC: dynamic x/y bins conditioned on each predicted region box."""

    def __init__(
        self,
        input_channels: int,
        num_landmarks: int = NUM_LANDMARKS,
        feature_channels: int = 128,
        num_bins: tuple[int, int] = (192, 256),
        sine_channels: int = 128,
    ):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.feature_channels = feature_channels
        self.pose_to_keypoints = nn.Sequential(
            nn.Linear(input_channels, num_landmarks * feature_channels),
            nn.LayerNorm(num_landmarks * feature_channels),
        )
        self.gau = GAUEncoder(feature_channels, attention_channels=feature_channels)
        self.keypoint_position = nn.Parameter(torch.randn(num_landmarks, feature_channels) * 0.02)
        self.sine = SinePositionEncoding(sine_channels)
        self.x_projection = nn.Linear(sine_channels, feature_channels)
        self.y_projection = nn.Linear(sine_channels, feature_channels)
        self.x_coordinate_scale = float(num_bins[0])
        self.y_coordinate_scale = float(num_bins[1])
        self.register_buffer("x_base", torch.linspace(-0.5, 0.5, num_bins[0]), persistent=False)
        self.register_buffer("y_base", torch.linspace(-0.5, 0.5, num_bins[1]), persistent=False)

    def forward(self, pose_features: torch.Tensor, boxes: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, regions, _ = pose_features.shape
        keypoint_features = self.pose_to_keypoints(pose_features).reshape(
            batch, regions, self.num_landmarks, self.feature_channels
        )
        keypoint_features = self.gau(keypoint_features, self.keypoint_position)
        center, scale = boxes.split(2, dim=-1)
        x_bins = self.x_base.view(1, 1, -1) * scale[..., 0:1] + center[..., 0:1]
        y_bins = self.y_base.view(1, 1, -1) * scale[..., 1:2] + center[..., 1:2]
        # The reference RTMO operates in pixel coordinates. Our boxes are
        # normalized, so scale them back to a bin-sized coordinate domain
        # before sine encoding to preserve useful positional frequencies.
        x_encoding = self.x_projection(self.sine(x_bins * self.x_coordinate_scale))
        y_encoding = self.y_projection(self.sine(y_bins * self.y_coordinate_scale))
        x_probability = torch.matmul(keypoint_features, x_encoding.transpose(-1, -2)).softmax(dim=-1)
        y_probability = torch.matmul(keypoint_features, y_encoding.transpose(-1, -2)).softmax(dim=-1)
        x = (x_probability * x_bins.unsqueeze(-2)).sum(dim=-1)
        y = (y_probability * y_bins.unsqueeze(-2)).sum(dim=-1)
        return {
            "coordinates": torch.stack((x, y), dim=-1).clamp(0.0, 1.0),
            "x_probability": x_probability,
            "y_probability": y_probability,
            "x_bins": x_bins,
            "y_bins": y_bins,
        }


class RTMOHeadModule(nn.Module):
    """Split classification and grouped pose branches from the RTMO head."""

    def __init__(
        self,
        feature_channels: int = 256,
        num_classes: int = 4,
        num_landmarks: int = NUM_LANDMARKS,
        groups: int = 8,
        channels_per_group: int = 16,
        pose_vector_channels: int = 256,
        levels: int = 2,
    ):
        super().__init__()
        if feature_channels % 2:
            raise ValueError("RTMO feature_channels must be even for classification/pose splitting")
        branch_input = feature_channels // 2
        pose_channels = groups * channels_per_group
        self.classification = nn.ModuleList(
            nn.Sequential(ConvNormAct(branch_input, feature_channels, 3), ConvNormAct(feature_channels, feature_channels, 3))
            for _ in range(levels)
        )
        self.pose = nn.ModuleList(
            nn.Sequential(
                ConvNormAct(branch_input, pose_channels, 3),
                ConvNormAct(pose_channels, pose_channels, 3, groups=groups),
                ConvNormAct(pose_channels, pose_channels, 3, groups=groups),
                ConvNormAct(pose_channels, pose_channels, 3, groups=groups),
            )
            for _ in range(levels)
        )
        self.out_class = nn.ModuleList(nn.Conv2d(feature_channels, num_classes, 1) for _ in range(levels))
        self.out_bbox = nn.ModuleList(nn.Conv2d(pose_channels, 4, 1) for _ in range(levels))
        self.out_offset = nn.ModuleList(nn.Conv2d(pose_channels, num_landmarks * 2, 1) for _ in range(levels))
        self.out_visibility = nn.ModuleList(nn.Conv2d(pose_channels, num_landmarks, 1) for _ in range(levels))
        self.out_pose = nn.ModuleList(nn.Conv2d(pose_channels, pose_vector_channels, 1) for _ in range(levels))

    def forward(self, features: tuple[torch.Tensor, ...]) -> tuple[list[torch.Tensor], ...]:
        classifications, boxes, offsets, visibility, pose_vectors = [], [], [], [], []
        for index, feature in enumerate(features):
            cls_input, pose_input = feature.chunk(2, dim=1)
            cls_feature = self.classification[index](cls_input)
            pose_feature = self.pose[index](pose_input)
            classifications.append(self.out_class[index](cls_feature))
            boxes.append(self.out_bbox[index](pose_feature))
            offsets.append(self.out_offset[index](pose_feature))
            visibility.append(self.out_visibility[index](pose_feature))
            pose_vectors.append(self.out_pose[index](pose_feature))
        return classifications, boxes, offsets, visibility, pose_vectors


def _flatten(values: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([value.flatten(2).transpose(1, 2) for value in values], dim=1)


class RTMOKneePose(nn.Module):
    """RTMO model specialized to one femur/tibia/fibula/patella per image."""

    def __init__(
        self,
        input_channels: int = 3,
        num_landmarks: int = NUM_LANDMARKS,
        backbone_width: int = 48,
        backbone_depth: int = 2,
        neck_channels: int = 256,
        attention_heads: int = 8,
        pose_vector_channels: int = 256,
        dcc_feature_channels: int = 128,
        dcc_bins: tuple[int, int] = (192, 256),
    ):
        super().__init__()
        if num_landmarks != NUM_LANDMARKS:
            raise ValueError(f"RTMO requires the canonical {NUM_LANDMARKS} landmarks")
        self.backbone = RTMOBackbone(input_channels, backbone_width, backbone_depth)
        self.neck = HybridEncoder(self.backbone.out_channels, neck_channels, attention_heads)
        self.head = RTMOHeadModule(
            feature_channels=neck_channels,
            num_landmarks=num_landmarks,
            pose_vector_channels=pose_vector_channels,
        )
        self.dcc = DynamicCoordinateClassifier(
            pose_vector_channels,
            num_landmarks,
            feature_channels=dcc_feature_channels,
            num_bins=dcc_bins,
            sine_channels=dcc_feature_channels,
        )

    @staticmethod
    def _grids(features: tuple[torch.Tensor, ...]) -> torch.Tensor:
        grids = []
        for feature in features:
            height, width = feature.shape[-2:]
            y, x = torch.meshgrid(
                (torch.arange(height, device=feature.device, dtype=feature.dtype) + 0.5) / height,
                (torch.arange(width, device=feature.device, dtype=feature.dtype) + 0.5) / width,
                indexing="ij",
            )
            grids.append(torch.stack((x, y), dim=-1).reshape(-1, 2))
        return torch.cat(grids, dim=0)

    def forward(self, images: torch.Tensor, return_aux: bool = False):
        features = self.neck(self.backbone(images))
        class_maps, box_maps, offset_maps, visibility_maps, pose_maps = self.head(features)
        class_logits = _flatten(class_maps)
        raw_boxes = _flatten(box_maps)
        raw_offsets = _flatten(offset_maps).reshape(images.shape[0], -1, NUM_LANDMARKS, 2)
        visibility_logits = _flatten(visibility_maps)
        pose_vectors = _flatten(pose_maps)
        grids = self._grids(features)

        region_weights = class_logits.transpose(1, 2).softmax(dim=-1)
        selected_pose = torch.einsum("brn,bnc->brc", region_weights, pose_vectors)
        selected_raw_boxes = torch.einsum("brn,bnd->brd", region_weights, raw_boxes)
        selected_offsets = torch.einsum("brn,bnkd->brkd", region_weights, raw_offsets)
        selected_visibility = torch.einsum("brn,bnk->brk", region_weights, visibility_logits)
        selected_grid = torch.einsum("brn,nd->brd", region_weights, grids)

        center = (selected_grid + selected_raw_boxes[..., :2].tanh() * 0.25).clamp(0.0, 1.0)
        scale = (selected_raw_boxes[..., 2:].sigmoid() * 0.9 + 0.1).clamp(0.05, 1.0)
        boxes = torch.cat((center, scale), dim=-1)
        dcc = self.dcc(selected_pose, boxes)
        proxy = (center[:, :, None] + selected_offsets.tanh() * scale[:, :, None] * 0.5).clamp(0.0, 1.0)
        region_scores = (region_weights * class_logits.transpose(1, 2).sigmoid()).sum(dim=-1)

        coordinates, proxy_chunks, visibility_chunks = [], [], []
        offset = 0
        visibility = selected_visibility.sigmoid()
        for class_id, count in enumerate(REGION_KEYPOINT_COUNTS):
            index = slice(offset, offset + count)
            coordinates.append(dcc["coordinates"][:, class_id, index])
            proxy_chunks.append(proxy[:, class_id, index])
            visibility_chunks.append(visibility[:, class_id, index])
            offset += count
        canonical = torch.cat(
            (torch.cat(coordinates, dim=1), torch.cat(visibility_chunks, dim=1).unsqueeze(-1)), dim=-1
        )
        if not return_aux:
            return canonical
        return {
            "canonical": canonical,
            "proxy_coordinates": torch.cat(proxy_chunks, dim=1),
            "boxes": boxes,
            "region_scores": region_scores,
            "dcc": dcc,
        }


__all__ = [
    "DynamicCoordinateClassifier",
    "GAUEncoder",
    "RTMOHeadModule",
    "RTMOKneePose",
    "ScaleNorm",
    "SinePositionEncoding",
]
