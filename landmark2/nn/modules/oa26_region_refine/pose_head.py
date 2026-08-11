# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Dedicated Pose26 integration for per-class MESKO4GF2 region refinement."""

from __future__ import annotations

import torch

from landmark2.nn.modules.oa26.pose_heads import OA26HeatmapPose
from landmark2.data.schema import (
    MAX_REGION_KEYPOINTS,
    NUM_REGIONS,
    validate_region_schema,
)
from landmark2.core.debug import debug_event
from landmark2.core.tal import make_anchors

from .refinement_head import OA26PerRegionRefinementHead


class OA26RegionRefinePose(OA26HeatmapPose):
    """Keep v1 paths and add an isolated P4 refiner for each of the four detected bones."""

    def __init__(
        self,
        nc: int = NUM_REGIONS,
        kpt_shape: tuple = (MAX_REGION_KEYPOINTS, 3),
        region_config: dict | None = None,
        reg_max: int = 1,
        end2end: bool = False,
        ch: tuple = (),
    ):
        """Initialize the unchanged v1 head and the v9-only region branch."""
        validate_region_schema(nc, kpt_shape)
        cfg = region_config or {}
        super().__init__(
            nc,
            kpt_shape,
            int(cfg.get("auxiliary_heatmap_channels", 0)),
            float(cfg.get("coarse_heatmap_temperature", 1.0)),
            reg_max,
            end2end,
            ch,
        )
        self.region_refine_enabled = bool(cfg.get("enabled", True))
        if str(cfg.get("feature_level", "P4")).upper() != "P4":
            raise ValueError("OA26 region refinement currently supports feature_level=P4 only")
        if int(cfg.get("num_regions", NUM_REGIONS)) != NUM_REGIONS:
            raise ValueError(f"MESKO4GF2 defines exactly {NUM_REGIONS} region classes")
        if len(ch) < 3:
            raise ValueError("OA26RegionRefinePose requires P2, P3, and P4 feature levels")
        self.region_refine_head = OA26PerRegionRefinementHead(
            in_channels=ch[2],
            num_classes=nc,
            kpt_shape=kpt_shape,
            d_model=int(cfg.get("d_model", 128)),
            num_heads=int(cfg.get("num_heads", 4)),
            num_layers=int(cfg.get("num_layers", 2)),
            roi_output_size=tuple(cfg.get("roi_output_size", (20, 20))),
            roi_sampling_ratio=int(cfg.get("roi_sampling_ratio", 2)),
            roi_padding=float(cfg.get("roi_padding", 0.25)),
            min_roi_size_px=float(cfg.get("min_roi_size_px", 48.0)),
            heatmap_temperature=float(cfg.get("heatmap_temperature", 0.1)),
            use_coarse_prior=bool(cfg.get("use_coarse_spatial_prior", False)),
            coarse_prior_sigma=float(cfg.get("coarse_prior_sigma", 0.25)),
            coarse_prior_gain=float(cfg.get("coarse_prior_gain", 0.5)),
            dropout=float(cfg.get("dropout", 0.1)),
            gradient_checkpointing=bool(cfg.get("gradient_checkpointing", False)),
        )
        if end2end:
            # Both E2E branches refine the same anatomy, so sharing avoids a redundant second transformer copy.
            self.one2one_region_refine_head = self.region_refine_head

    @property
    def one2many(self):
        """Return v1 one-to-many modules plus the separate v9 refiner."""
        heads = super().one2many
        # Evaluation publishes the one-to-one branch, so avoid refining an unused duplicate prediction set.
        heads["region_refine_head"] = self.region_refine_head if self.training else None
        return heads

    @property
    def one2one(self):
        """Return v1 one-to-one modules plus the separate v9 refiner."""
        heads = super().one2one
        heads["region_refine_head"] = self.one2one_region_refine_head
        return heads

    def _select_class_instances(
        self,
        x: list[torch.Tensor],
        boxes: torch.Tensor,
        scores: torch.Tensor,
        raw_kpts: torch.Tensor,
        image_size: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select and decode the highest-confidence anchor for every class in every image."""
        batch_size = boxes.shape[0]
        image_h, image_w = image_size
        selected = scores.sigmoid().argmax(dim=2)  # B x C; one region instance per MESKO class.
        batch_grid = torch.arange(batch_size, device=boxes.device)[:, None].expand(-1, self.nc)
        class_grid = torch.arange(self.nc, device=boxes.device)[None].expand(batch_size, -1)

        if bool((self.stride > 0).all()):
            anchors, stride_tensor = make_anchors(x, self.stride, 0.5)
            # Gather B x C selected anchors before decoding. Decoding B x A x K for all 66k+ anchors at imgsz=896
            # created multi-GB temporary tensors at realistic batch sizes and could make a notebook kernel get killed.
            box_index = selected[:, None].expand(-1, boxes.shape[1], -1)
            selected_boxes = boxes.gather(2, box_index)
            selected_anchors = anchors[selected]
            selected_stride = stride_tensor[selected]
            instance_boxes = self.decode_bboxes(selected_boxes, selected_anchors.transpose(1, 2))
            instance_boxes = (instance_boxes * selected_stride.transpose(1, 2)).transpose(1, 2)

            kpt_index = selected[:, None].expand(-1, self.nk, -1)
            selected_raw = raw_kpts.gather(2, kpt_index).permute(0, 2, 1)
            selected_raw = selected_raw.reshape(batch_size, self.nc, *self.kpt_shape)
            decoded_xy = (selected_raw[..., :2] + selected_anchors[:, :, None]) * selected_stride[:, :, None]
            decoded_conf = selected_raw[..., 2:3].sigmoid()
            coarse_kpts = torch.cat((decoded_xy, decoded_conf), dim=-1)
        else:
            # PoseModel runs a stride-discovery forward before prediction heads are calibrated.
            full = boxes.new_tensor((0.0, 0.0, float(image_w), float(image_h)))
            instance_boxes = full.view(1, 1, 4).expand(batch_size, self.nc, -1)
            coarse_kpts = raw_kpts.new_zeros((batch_size, self.nc, *self.kpt_shape))
            coarse_kpts[..., 0] = image_w * 0.5
            coarse_kpts[..., 1] = image_h * 0.5
            coarse_kpts[..., 2] = 0.5

        instance_boxes = instance_boxes.reshape(-1, 4)
        x1 = instance_boxes[:, 0].clamp(0, image_w)
        y1 = instance_boxes[:, 1].clamp(0, image_h)
        x2 = instance_boxes[:, 2].clamp(0, image_w)
        y2 = instance_boxes[:, 3].clamp(0, image_h)
        x2 = torch.maximum(x2, x1 + 2.0).clamp(max=image_w)
        y2 = torch.maximum(y2, y1 + 2.0).clamp(max=image_h)
        instance_boxes = torch.stack((x1, y1, x2, y2), dim=-1)
        return (
            instance_boxes,
            coarse_kpts.reshape(-1, *self.kpt_shape),
            class_grid.reshape(-1),
            batch_grid.reshape(-1),
            selected,
        )

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: torch.nn.Module,
        cls_head: torch.nn.Module,
        pose_head: torch.nn.Module,
        kpts_head: torch.nn.Module,
        kpts_sigma_head: torch.nn.Module,
        heatmap_head: torch.nn.Module | None = None,
        region_refine_head: torch.nn.Module | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return standard v1 predictions plus one independent refinement row per class."""
        preds = super().forward_head(x, box_head, cls_head, pose_head, kpts_head, kpts_sigma_head, heatmap_head)
        if not self.region_refine_enabled or region_refine_head is None or "kpts" not in preds:
            return preds

        stride0 = self.stride[0].to(device=x[0].device, dtype=x[0].dtype)
        stride0 = torch.where(stride0 > 0, stride0, stride0.new_tensor(4.0))
        image_h = x[0].new_tensor(float(x[0].shape[-2])) * stride0
        image_w = x[0].new_tensor(float(x[0].shape[-1])) * stride0
        boxes, coarse, class_ids, batch_ids, selected = self._select_class_instances(
            x, preds["boxes"], preds["scores"], preds["kpts"], (image_h, image_w)
        )
        p4_stride = self.stride[2] if self.stride[2] > 0 else self.stride.new_tensor(16.0)
        preds.update(
            region_refine_head(
                x[2], boxes, coarse, class_ids, batch_ids, selected, (image_h, image_w), p4_stride
            )
        )
        return preds

    def _postprocess_refined(self, preds: torch.Tensor, raw: dict[str, torch.Tensor]) -> torch.Tensor:
        """Postprocess standard anchors first, then inject only the matching compact class refinement."""
        boxes, scores, coarse_kpts = preds.split([4, self.nc, self.nk], dim=-1)
        scores, classes, indices = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(1, indices.expand(-1, -1, 4))
        coarse_kpts = coarse_kpts.gather(1, indices.expand(-1, -1, self.nk))
        batch_size, detections = indices.shape[:2]
        if "refined_region_kpts" not in raw:
            return torch.cat((boxes, scores, classes, coarse_kpts), dim=-1)

        class_index = classes.long().squeeze(-1)
        refined = raw["refined_region_kpts"].reshape(batch_size, self.nc, self.nk)
        refined = refined.gather(1, class_index[:, :, None].expand(-1, -1, self.nk))
        selected = raw["region_selected_anchor_indices"].gather(1, class_index)
        use_refined = indices.squeeze(-1).eq(selected).unsqueeze(-1)
        kpts = torch.where(use_refined, refined, coarse_kpts)
        return torch.cat((boxes, scores, classes, kpts), dim=-1)

    def forward(
        self, x: list[torch.Tensor]
    ) -> dict[str, torch.Tensor] | torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run v9 without expanding refined poses across every class-anchor combination."""
        debug_event("v9-head-forward-enter", training=self.training, batch=x[0].shape[0], p2=tuple(x[0].shape))
        preds = self.forward_head(x, **self.one2many)
        debug_event("v9-head-one2many-complete")
        if self.end2end:
            x_detach = [feature.detach() for feature in x]
            one2one = self.forward_head(x_detach, **self.one2one)
            debug_event("v9-head-one2one-complete")
            preds = {"one2many": preds, "one2one": one2one}
        if self.training:
            debug_event("v9-head-forward-complete", output="train")
            return preds

        raw = preds["one2one"] if self.end2end else preds
        decoded = super()._inference(raw)
        if self.end2end:
            decoded = self._postprocess_refined(decoded.permute(0, 2, 1), raw)
        debug_event("v9-head-forward-complete", output="eval", decoded_shape=tuple(decoded.shape))
        return decoded if self.export else (decoded, preds)

    def fuse(self) -> None:
        """Discard only training-time one-to-many v9 modules during inference fusion."""
        super().fuse()
        self.region_refine_head = None
