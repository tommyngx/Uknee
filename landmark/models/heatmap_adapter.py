"""Ultralytics lifecycle adapters for canonical HRNet, ViTPose and RTMO models."""

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.nn import functional as F

from landmark.core.model import Model
from landmark.core.predictor import BasePredictor
from landmark.core.results import Results
from landmark.core.detect import DetectionTrainer
from landmark.core import DEFAULT_CFG_DICT, LOGGER, RANK, ops
from landmark.core.targets import extract_canonical_image_keypoints, gaussian_heatmap_targets
from landmark.core.torch_utils import model_info

from landmark.data.schema import NUM_LANDMARKS, REGION_KEYPOINT_COUNTS, REGION_NAMES
from landmark.models import HRNetW32, HRNetW48, RTMOKneePose, ViTPoseB, ViTPoseS
from landmark.utils.validation import FlatPoseTrainerMixin, KneePoseValidator


def _read_config(config: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config, dict):
        return dict(config)
    path = Path(config)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data["yaml_file"] = str(path)
    return data


def _soft_argmax(heatmaps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch, keypoints, height, width = heatmaps.shape
    probabilities = heatmaps.flatten(2).softmax(dim=-1).view(batch, keypoints, height, width)
    xs = torch.linspace(0.0, 1.0, width, device=heatmaps.device, dtype=heatmaps.dtype)
    ys = torch.linspace(0.0, 1.0, height, device=heatmaps.device, dtype=heatmaps.dtype)
    x = (probabilities.sum(dim=2) * xs).sum(dim=-1)
    y = (probabilities.sum(dim=3) * ys).sum(dim=-1)
    confidence = heatmaps.flatten(2).amax(dim=-1).sigmoid()
    return torch.stack((x, y), dim=-1), confidence


def canonical_to_objects(canonical: torch.Tensor, image_hw: tuple[int, int]) -> list[dict[str, torch.Tensor]]:
    """Convert normalized canonical predictions to four fixed pose objects."""
    height, width = image_hw
    output: list[dict[str, torch.Tensor]] = []
    for sample in canonical:
        boxes, scores, classes, keypoints = [], [], [], []
        offset = 0
        for class_id, count in enumerate(REGION_KEYPOINT_COUNTS):
            local = sample.new_zeros((51, 3))
            local[:count] = sample[offset : offset + count]
            local[:count, 0] *= width
            local[:count, 1] *= height
            points = local[:count, :2]
            x1y1, x2y2 = points.amin(dim=0), points.amax(dim=0)
            boxes.append(torch.cat((x1y1, x2y2)))
            scores.append(local[:count, 2].mean())
            classes.append(sample.new_tensor(float(class_id)))
            keypoints.append(local)
            offset += count
        output.append(
            {
                "bboxes": torch.stack(boxes),
                "conf": torch.stack(scores),
                "cls": torch.stack(classes),
                "keypoints": torch.stack(keypoints),
            }
        )
    return output


class HeatmapPoseModel(nn.Module):
    """Full-frame heatmap model satisfying the vendored trainer contract."""

    def __init__(self, config: str | Path | dict[str, Any], verbose: bool = False):
        super().__init__()
        self.yaml = _read_config(config)
        self.yaml_file = self.yaml.get("yaml_file", "heatmap-pose.yaml")
        self.architecture = str(self.yaml["architecture"])
        common = {
            "input_channels": int(self.yaml.get("input_channels", 3)),
            "num_landmarks": int(self.yaml.get("num_landmarks", NUM_LANDMARKS)),
        }
        if common["num_landmarks"] != NUM_LANDMARKS:
            raise ValueError(f"Canonical models require {NUM_LANDMARKS} landmarks")
        if self.architecture == "hrnet_w32":
            self.network = HRNetW32(**common)
        elif self.architecture == "hrnet_w48":
            self.network = HRNetW48(**common)
        elif self.architecture in {"vitpose_s", "vitpose_b"}:
            vitpose_class = ViTPoseS if self.architecture == "vitpose_s" else ViTPoseB
            self.network = vitpose_class(
                **common,
                image_size=int(self.yaml.get("image_size", 640)),
                patch_size=int(self.yaml.get("patch_size", 16)),
                embed_dim=int(self.yaml.get("embed_dim", 384 if self.architecture == "vitpose_s" else 768)),
                depth=int(self.yaml.get("depth", 12)),
                attention_heads=int(self.yaml.get("attention_heads", 6 if self.architecture == "vitpose_s" else 12)),
                mlp_ratio=int(self.yaml.get("mlp_ratio", 4)),
                dropout=float(self.yaml.get("dropout", 0.0)),
            )
        elif self.architecture == "rtmo":
            self.network = RTMOKneePose(
                **common,
                backbone_width=int(self.yaml.get("backbone_width", 48)),
                backbone_depth=int(self.yaml.get("backbone_depth", 2)),
                neck_channels=int(self.yaml.get("neck_channels", 256)),
                attention_heads=int(self.yaml.get("attention_heads", 8)),
                pose_vector_channels=int(self.yaml.get("pose_vector_channels", 256)),
                dcc_feature_channels=int(self.yaml.get("dcc_feature_channels", 128)),
                dcc_bins=tuple(self.yaml.get("dcc_bins", (192, 256))),
            )
        else:
            raise ValueError(f"Unsupported canonical architecture: {self.architecture}")
        self.nc = 4
        self.names = dict(enumerate(REGION_NAMES))
        self.kpt_shape = [51, 3]
        self.stride = torch.tensor([32.0])
        self.task = "pose"
        self.args = {**DEFAULT_CFG_DICT, "task": "pose", "model": self.yaml_file}
        self.criterion = None
        if verbose:
            self.info()

    def forward(self, inputs: torch.Tensor | dict[str, torch.Tensor], *args, **kwargs):
        if isinstance(inputs, dict):
            return self.loss(inputs)
        output = self.network(inputs)
        if self.architecture == "rtmo":
            return output
        heatmaps = output
        if self.training:
            return heatmaps
        coordinates, confidence = _soft_argmax(heatmaps)
        return torch.cat((coordinates, confidence.unsqueeze(-1)), dim=-1)

    def loss(self, batch: dict[str, torch.Tensor], preds: torch.Tensor | None = None):
        if self.architecture == "rtmo":
            return self._rtmo_loss(batch)
        heatmaps = preds if isinstance(preds, torch.Tensor) and preds.ndim == 4 else self.network(batch["img"])
        batch_size, _, heatmap_height, heatmap_width = heatmaps.shape
        image_size = heatmaps.new_tensor(batch["img"].shape[-2:])
        ground_truth, visible = extract_canonical_image_keypoints(
            batch, batch_size, image_size, heatmaps.device, heatmaps.dtype
        )
        target = gaussian_heatmap_targets(
            ground_truth,
            visible,
            (heatmap_height, heatmap_width),
            image_size,
            float(self.yaml.get("heatmap_sigma", 1.5)),
        )
        mask = visible[:, :, None, None].to(heatmaps.dtype)
        denominator = (mask.sum() * heatmap_height * heatmap_width).clamp(min=1)
        heatmap_loss = ((heatmaps.sigmoid() - target).square() * mask).sum() / denominator

        normalized_target = ground_truth / image_size.flip(0).view(1, 1, 2).clamp(min=1)
        predicted_coordinates, _ = _soft_argmax(heatmaps)
        coordinate_loss = F.smooth_l1_loss(
            predicted_coordinates[visible], normalized_target[visible], beta=0.01
        ) if visible.any() else heatmap_loss.new_zeros(())
        coordinate_gain = float(self.yaml.get("coordinate_loss_gain", 1.0))
        total = heatmap_loss + coordinate_gain * coordinate_loss
        items = torch.stack((heatmap_loss.detach(), coordinate_loss.detach()))
        return total * batch_size, items

    def _rtmo_loss(self, batch: dict[str, torch.Tensor]):
        predictions = self.network(batch["img"], return_aux=True)
        canonical = predictions["canonical"]
        batch_size = canonical.shape[0]
        image_size = canonical.new_tensor(batch["img"].shape[-2:])
        ground_truth, visible = extract_canonical_image_keypoints(
            batch, batch_size, image_size, canonical.device, canonical.dtype
        )
        target = ground_truth / image_size.flip(0).view(1, 1, 2).clamp(min=1)
        coordinate_loss = F.smooth_l1_loss(
            canonical[..., :2][visible], target[visible], beta=0.01
        ) if visible.any() else canonical.sum() * 0.0
        proxy_loss = F.smooth_l1_loss(
            predictions["proxy_coordinates"][visible], target[visible], beta=0.01
        ) if visible.any() else canonical.sum() * 0.0
        visibility_loss = F.binary_cross_entropy(
            canonical[..., 2].clamp(1e-5, 1.0 - 1e-5), visible.to(canonical.dtype)
        )

        target_boxes, region_present = [], []
        offset = 0
        for count in REGION_KEYPOINT_COUNTS:
            region_target = target[:, offset : offset + count]
            region_visible = visible[:, offset : offset + count]
            minimum = torch.where(region_visible[..., None], region_target, torch.inf).amin(dim=1)
            maximum = torch.where(region_visible[..., None], region_target, -torch.inf).amax(dim=1)
            present = region_visible.any(dim=1)
            minimum = torch.where(present[:, None], minimum, torch.zeros_like(minimum))
            maximum = torch.where(present[:, None], maximum, torch.zeros_like(maximum))
            target_boxes.append(torch.cat(((minimum + maximum) * 0.5, maximum - minimum), dim=-1))
            region_present.append(present)
            offset += count
        target_boxes = torch.stack(target_boxes, dim=1)
        region_present = torch.stack(region_present, dim=1)
        bbox_loss = F.smooth_l1_loss(
            predictions["boxes"][region_present], target_boxes[region_present], beta=0.02
        ) if region_present.any() else canonical.sum() * 0.0
        classification_loss = F.binary_cross_entropy(
            predictions["region_scores"].clamp(1e-5, 1.0 - 1e-5), region_present.to(canonical.dtype)
        )

        mle_terms = []
        dcc = predictions["dcc"]
        offset = 0
        for class_id, count in enumerate(REGION_KEYPOINT_COUNTS):
            index = slice(offset, offset + count)
            region_target = target[:, index]
            region_visible = visible[:, index]
            x_index = (dcc["x_bins"][:, class_id, None] - region_target[..., 0, None]).abs().argmin(dim=-1)
            y_index = (dcc["y_bins"][:, class_id, None] - region_target[..., 1, None]).abs().argmin(dim=-1)
            x_probability = dcc["x_probability"][:, class_id, index].gather(-1, x_index[..., None]).squeeze(-1)
            y_probability = dcc["y_probability"][:, class_id, index].gather(-1, y_index[..., None]).squeeze(-1)
            probability = (x_probability * y_probability).clamp_min(1e-9)
            if region_visible.any():
                mle_terms.append(-probability[region_visible].log().mean())
            offset += count
        mle_loss = torch.stack(mle_terms).mean() if mle_terms else canonical.sum() * 0.0
        dcc_loss = mle_loss + classification_loss

        total = (
            float(self.yaml.get("coordinate_loss_gain", 5.0)) * coordinate_loss
            + float(self.yaml.get("proxy_loss_gain", 1.0)) * proxy_loss
            + float(self.yaml.get("visibility_loss_gain", 1.0)) * visibility_loss
            + float(self.yaml.get("bbox_loss_gain", 2.0)) * bbox_loss
            + float(self.yaml.get("mle_loss_gain", 1.0)) * dcc_loss
        )
        items = torch.stack(
            tuple(value.detach() for value in (coordinate_loss, proxy_loss, bbox_loss, visibility_loss, dcc_loss))
        )
        return total * batch_size, items

    def load(self, weights: nn.Module | dict[str, torch.Tensor], verbose: bool = True):
        source = weights.state_dict() if isinstance(weights, nn.Module) else weights
        incompatible = self.load_state_dict(source, strict=False)
        if verbose and (incompatible.missing_keys or incompatible.unexpected_keys):
            LOGGER.warning(
                "Heatmap weights loaded non-strictly: %d missing, %d unexpected",
                len(incompatible.missing_keys),
                len(incompatible.unexpected_keys),
            )
        return self

    def fuse(self, verbose: bool = False):
        return self

    def is_fused(self, thresh: int = 10) -> bool:
        return False

    def info(self, detailed: bool = False, verbose: bool = True, imgsz: int = 640):
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)


class HeatmapPosePredictor(BasePredictor):
    """Turn canonical heatmap output into ordinary Ultralytics Results."""

    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        from landmark.core import DEFAULT_CFG

        super().__init__(cfg or DEFAULT_CFG, overrides, _callbacks)
        self.args.task = "pose"

    def postprocess(self, preds, img, orig_imgs):
        canonical = preds[0] if isinstance(preds, (tuple, list)) else preds
        objects = canonical_to_objects(canonical, tuple(img.shape[-2:]))
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]
        results = []
        for prediction, orig_img, image_path in zip(objects, orig_imgs, self.batch[0]):
            boxes = ops.scale_boxes(img.shape[-2:], prediction["bboxes"].clone(), orig_img.shape)
            keypoints = ops.scale_coords(img.shape[-2:], prediction["keypoints"].clone(), orig_img.shape)
            box_rows = torch.cat(
                (boxes, prediction["conf"][:, None], prediction["cls"][:, None]), dim=1
            )
            results.append(
                Results(orig_img, path=image_path, names=dict(enumerate(REGION_NAMES)), boxes=box_rows, keypoints=keypoints)
            )
        return results


class HeatmapPoseValidator(KneePoseValidator):
    """Evaluate fixed four-object heatmap predictions with pose and medical metrics."""

    def postprocess(self, preds: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        canonical = preds[0] if isinstance(preds, (tuple, list)) else preds
        return canonical_to_objects(canonical, tuple(self.batch["img"].shape[-2:]))

    def __call__(self, *args, **kwargs):
        self.batch = None
        return super().__call__(*args, **kwargs)

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        result = super().preprocess(batch)
        self.batch = result
        return result


class HeatmapPoseTrainer(FlatPoseTrainerMixin, DetectionTrainer):
    """Reuse upstream AMP/EMA/optimizer/scheduler/DDP for heatmap baselines."""

    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        from landmark.core import DEFAULT_CFG

        values = dict(overrides or {})
        values.update(task="pose", mosaic=0.0, mixup=0.0, cutmix=0.0)
        super().__init__(cfg or DEFAULT_CFG, values, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        model = HeatmapPoseModel(cfg, verbose=verbose and RANK == -1)
        if weights is not None:
            model.load(weights)
        return model

    def set_model_attributes(self):
        self.model.nc = self.data["nc"]
        self.model.names = self.data["names"]
        self.model.kpt_shape = self.data["kpt_shape"]
        self.model.args = self.args

    def get_dataset(self):
        data = super().get_dataset()
        if data.get("kpt_shape") != [51, 3]:
            raise ValueError("Heatmap baselines require kpt_shape=[51, 3]")
        return data

    def get_validator(self):
        if getattr(self.model, "architecture", "") == "rtmo":
            self.loss_names = ("coord_loss", "proxy_loss", "bbox_loss", "vis_loss", "dcc_loss")
        else:
            self.loss_names = ("heatmap_loss", "coord_loss")
        return HeatmapPoseValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )


class HeatmapPose(Model):
    """Ultralytics-style Python API for canonical full-frame models."""

    def __init__(self, model: str | Path, verbose: bool = False):
        super().__init__(model=model, task="pose", verbose=verbose)

    @property
    def task_map(self):
        return {
            "pose": {
                "model": HeatmapPoseModel,
                "trainer": HeatmapPoseTrainer,
                "validator": HeatmapPoseValidator,
                "predictor": HeatmapPosePredictor,
            }
        }


__all__ = [
    "HeatmapPose",
    "HeatmapPoseModel",
    "HeatmapPosePredictor",
    "HeatmapPoseTrainer",
    "HeatmapPoseValidator",
    "canonical_to_objects",
]
