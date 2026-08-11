"""Knee-specific validation, flat checkpoints and medical reporting."""

from __future__ import annotations

import csv
import random
import shutil
from copy import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from landmark2.core.pose import PoseTrainer, PoseValidator
from landmark2.core import LOGGER, RANK, ops
from landmark2.core.metrics import PoseMetrics

from landmark2.data.schema import LANDMARK_PATH_RANGES, REGION_KEYPOINT_COUNTS, REGION_NAMES
from landmark2.core.plotting import plot_dashboard_pose, plot_pose_metrics, plot_validation_samples


PIXEL_SPACING_MM = 0.10
RESULT_COLUMNS = (
    "epoch",
    "train/loss",
    "val/loss",
    "metrics/MRE",
    "metrics/PCK2",
    "metrics/PCK4",
    "metrics/PCK8",
    "metrics/HD95",
    "metrics/mAP50-95(B)",
    *(f"metrics/MRE_{name}" for name in REGION_NAMES),
)


def _to_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _sum_losses(metrics: dict[str, Any], prefix: str) -> float:
    values = [_to_float(value) for key, value in metrics.items() if key.startswith(prefix) and key.endswith("loss")]
    finite = [value for value in values if np.isfinite(value)]
    return float(sum(finite)) if finite else float("nan")


class FlatPoseTrainerMixin:
    """Keep checkpoints in the run root and select best.pt by minimum MRE."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        old_weights = self.save_dir / "weights"
        self.wdir = self.save_dir
        self.last = self.save_dir / "last.pt"
        self.best = self.save_dir / "best.pt"
        self.save_period = self.args.save_period = -1
        if RANK in {-1, 0}:
            self.callbacks["on_fit_epoch_end"] = [
                callback
                for callback in self.callbacks["on_fit_epoch_end"]
                if callback.__name__ != "plot_v9_performance_on_epoch_end"
            ]
            for pattern in (
                "pose_detection_performance.png",
                "results.png",
                "labels.jpg",
                "val_batch*.jpg",
                "*_curve.png",
                "confusion_matrix*.png",
                "best_mre.pt",
                "best_mre.txt",
                "dataset_resolved.yaml",
            ):
                for artifact in self.save_dir.glob(pattern):
                    artifact.unlink(missing_ok=True)
            for stale_dir in (self.save_dir / "labels", self.save_dir / "visualizations"):
                if stale_dir.exists():
                    shutil.rmtree(stale_dir)
            if old_weights.exists() and not self.args.resume:
                shutil.rmtree(old_weights)

    def validate(self):
        previous_best = self.best_fitness
        metrics, _ = super().validate()
        self.best_fitness = previous_best
        if metrics is None:
            return None, None
        mre = _to_float(metrics.get("metrics/MRE"))
        if np.isfinite(mre):
            fitness = -mre
        else:
            fitness = -_sum_losses(metrics, "val/")
        if self.best_fitness is None or fitness > self.best_fitness:
            self.best_fitness = fitness
        return metrics, fitness

    def save_metrics(self, metrics: dict[str, Any]) -> None:
        """Write one compact, stable row and refresh dashboard_pose.png."""
        row = {
            "epoch": self.epoch + 1,
            "train/loss": _sum_losses(metrics, "train/"),
            "val/loss": _sum_losses(metrics, "val/"),
        }
        row.update({key: _to_float(metrics.get(key)) for key in RESULT_COLUMNS if key not in row and key != "epoch"})
        self.csv.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.csv.exists()
        with self.csv.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=RESULT_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        plot_dashboard_pose(self.csv, self.save_dir / "dashboard_pose.png", pixel_spacing=PIXEL_SPACING_MM)


MEDICAL_KEYS = (
    "metrics/MRE",
    "metrics/median_error",
    "metrics/p95_error",
    "metrics/PCK2",
    "metrics/PCK4",
    "metrics/PCK8",
    "metrics/HD95",
    "metrics/BoxIoU",
    "metrics/failure_rate",
    "metrics/order_accuracy",
    "metrics/topology_failure_rate",
    *(f"metrics/MRE_{name}" for name in REGION_NAMES),
)


class UkneePoseMetrics(PoseMetrics):
    """PoseMetrics with scalar medical landmark measurements."""

    def __init__(self, names: dict[int, str] | None = None) -> None:
        super().__init__(names)
        self.medical = {key: 0.0 for key in MEDICAL_KEYS}

    @property
    def keys(self) -> list[str]:
        return [*super().keys, *MEDICAL_KEYS]

    def mean_results(self) -> list[float]:
        return [*super().mean_results(), *(self.medical[key] for key in MEDICAL_KEYS)]

    @property
    def fitness(self) -> float:
        return super().fitness


def _box_iou(left: torch.Tensor, right: torch.Tensor) -> float:
    intersection_xy = torch.minimum(left[2:], right[2:]) - torch.maximum(left[:2], right[:2])
    intersection = intersection_xy.clamp(min=0).prod()
    left_area = (left[2:] - left[:2]).clamp(min=0).prod()
    right_area = (right[2:] - right[:2]).clamp(min=0).prod()
    return float(intersection / (left_area + right_area - intersection).clamp(min=1e-9))


class KneePoseValidator(PoseValidator):
    """Add MRE/PCK/HD95/topology metrics and standardized plots."""

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.metrics = UkneePoseMetrics()
        self._medical_records: list[dict[str, Any]] = []
        self._sample_records: dict[str, dict[str, Any]] = {}
        self._sample_paths: list[str] = []
        self.current_epoch = 0

    def __call__(self, trainer=None, model=None):
        self.current_epoch = trainer.epoch + 1 if trainer is not None else 0
        return super().__call__(trainer=trainer, model=model)

    def init_metrics(self, model: torch.nn.Module) -> None:
        super().init_metrics(model)
        self._medical_records.clear()
        self._sample_records.clear()
        dataset = getattr(self.dataloader, "dataset", None)
        paths = sorted(str(Path(path).resolve()) for path in getattr(dataset, "im_files", ()))
        rng = random.Random(2006)
        self._sample_paths = rng.sample(paths, min(4, len(paths))) if paths else []

    def update_metrics(self, preds: list[dict[str, torch.Tensor]], batch: dict[str, Any]) -> None:
        super().update_metrics(preds, batch)
        for image_index, prediction in enumerate(preds):
            if not self.args.plots:
                pbatch = self._prepare_batch(image_index, batch)
                self.confusion_matrix.process_batch(self._prepare_pred(prediction), pbatch, conf=self.args.conf)
            self._record_medical(image_index, prediction, batch)

    def _record_medical(self, image_index: int, prediction: dict[str, torch.Tensor], batch: dict[str, Any]) -> None:
        pbatch = self._prepare_batch(image_index, batch)
        classes = pbatch["cls"].long()
        diagonal = float(np.hypot(*pbatch["ori_shape"]))
        total_points = sum(REGION_KEYPOINT_COUNTS)
        pred_canonical = np.full((total_points, 2), np.nan, dtype=np.float32)
        gt_canonical = np.full_like(pred_canonical, np.nan)
        pred_input = np.full_like(pred_canonical, np.nan)
        valid_canonical = np.zeros(total_points, dtype=bool)
        all_errors: list[np.ndarray] = []
        region_errors: dict[int, np.ndarray] = {}
        box_ious: list[float] = []
        offset = 0

        for class_id, count in enumerate(REGION_KEYPOINT_COUNTS):
            target_rows = torch.nonzero(classes == class_id, as_tuple=False).flatten()
            pred_rows = torch.nonzero(prediction["cls"].long() == class_id, as_tuple=False).flatten()
            target_row = int(target_rows[0]) if target_rows.numel() else None
            selected = int(pred_rows[prediction["conf"][pred_rows].argmax()]) if pred_rows.numel() else None
            if target_row is None:
                offset += count
                continue
            target_input = pbatch["keypoints"][target_row, :count].clone()
            visible = target_input[:, 2] > 0 if target_input.shape[-1] == 3 else torch.ones(
                count, dtype=torch.bool, device=target_input.device
            )
            valid_canonical[offset : offset + count] = visible.detach().cpu().numpy()
            target = ops.scale_coords(
                pbatch["imgsz"], target_input.clone(), pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"]
            )
            gt_xy = target[:, :2]
            gt_canonical[offset : offset + count] = gt_xy.detach().cpu().numpy()
            if selected is None:
                errors = np.full(int(visible.sum()), diagonal, dtype=np.float32)
                box_ious.append(0.0)
            else:
                predicted_input = prediction["keypoints"][selected, :count].clone()
                pred_input[offset : offset + count] = predicted_input[:, :2].detach().cpu().numpy()
                predicted = ops.scale_coords(
                    pbatch["imgsz"], predicted_input, pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"]
                )
                pred_xy = predicted[:, :2]
                pred_canonical[offset : offset + count] = pred_xy.detach().cpu().numpy()
                errors = torch.linalg.vector_norm(pred_xy[visible] - gt_xy[visible], dim=-1).detach().cpu().numpy()
                box_ious.append(_box_iou(pbatch["bboxes"][target_row], prediction["bboxes"][selected]))
            region_errors[class_id] = errors
            all_errors.append(errors)
            offset += count

        if not all_errors:
            return
        errors = np.concatenate(all_errors)
        order_values: list[bool] = []
        topology_failures: list[bool] = []
        for start, stop in LANDMARK_PATH_RANGES:
            pred_path, gt_path = pred_canonical[start:stop], gt_canonical[start:stop]
            valid = np.isfinite(pred_path).all(1) & np.isfinite(gt_path).all(1) & valid_canonical[start:stop]
            if valid.sum() < 2:
                topology_failures.append(True)
                continue
            agreement = np.sum(np.diff(pred_path[valid], axis=0) * np.diff(gt_path[valid], axis=0), axis=1) >= 0
            order_values.extend(agreement.tolist())
            topology_failures.append(not bool(np.all(agreement)))
        valid_gt = valid_canonical & np.isfinite(gt_canonical).all(axis=1)
        valid_pred = valid_gt & np.isfinite(pred_canonical).all(axis=1)
        if valid_pred.any():
            distances = np.linalg.norm(
                pred_canonical[valid_pred, None, :] - gt_canonical[valid_gt][None, :, :], axis=-1
            )
            hausdorff_values = np.concatenate((distances.min(axis=1), distances.min(axis=0)))
            missing = int(valid_gt.sum() - valid_pred.sum())
            if missing:
                hausdorff_values = np.concatenate((hausdorff_values, np.full(missing, diagonal)))
            hd95 = float(np.percentile(hausdorff_values, 95))
        else:
            hd95 = diagonal
        record = {
            "errors": errors,
            "regions": region_errors,
            "hd95": hd95,
            "box_iou": float(np.mean(box_ious)) if box_ious else 0.0,
            "failed": bool(errors.mean() > 8.0 or np.any(errors >= diagonal)),
            "order": order_values,
            "topology": topology_failures,
        }
        self._medical_records.append(record)

        resolved = str(Path(pbatch["im_file"]).resolve())
        if resolved in self._sample_paths:
            image = batch["img"][image_index].detach().float().cpu().numpy().transpose(1, 2, 0)
            image = np.clip(image, 0, 1)
            self._sample_records[resolved] = {
                "path": resolved,
                "image": image,
                "pred": pred_input,
                "valid": valid_canonical,
                "mre_px": float(errors.mean()),
                "pck2": float((errors <= 2).mean()),
                "hd95_px": record["hd95"],
                "box_iou": record["box_iou"],
            }

    def gather_stats(self) -> None:
        super().gather_stats()
        payload = {"medical": self._medical_records, "samples": self._sample_records}
        if RANK == 0:
            gathered: list[Any] = [None] * dist.get_world_size()
            dist.gather_object(payload, gathered, dst=0)
            self._medical_records = [record for item in gathered for record in (item or {}).get("medical", [])]
            self._sample_records = {
                path: record for item in gathered for path, record in (item or {}).get("samples", {}).items()
            }
        elif RANK > 0:
            dist.gather_object(payload, None, dst=0)
            self._medical_records.clear()
            self._sample_records.clear()

    def get_stats(self) -> dict[str, Any]:
        stats = super().get_stats()
        records = self._medical_records
        if not records:
            return stats
        errors = np.concatenate([record["errors"] for record in records])
        order = [value for record in records for value in record["order"]]
        topology = [value for record in records for value in record["topology"]]
        values = {
            "metrics/MRE": float(errors.mean()),
            "metrics/median_error": float(np.median(errors)),
            "metrics/p95_error": float(np.percentile(errors, 95)),
            "metrics/PCK2": float((errors <= 2).mean()),
            "metrics/PCK4": float((errors <= 4).mean()),
            "metrics/PCK8": float((errors <= 8).mean()),
            "metrics/HD95": float(np.mean([record["hd95"] for record in records])),
            "metrics/BoxIoU": float(np.mean([record["box_iou"] for record in records])),
            "metrics/failure_rate": float(np.mean([record["failed"] for record in records])),
            "metrics/order_accuracy": float(np.mean(order)) if order else float("nan"),
            "metrics/topology_failure_rate": float(np.mean(topology)) if topology else float("nan"),
        }
        for class_id, name in enumerate(REGION_NAMES):
            region = [record["regions"][class_id] for record in records if class_id in record["regions"]]
            values[f"metrics/MRE_{name}"] = float(np.concatenate(region).mean()) if region else float("nan")
        self.metrics.medical.update(values)
        stats.update(values)
        return stats

    def finalize_metrics(self) -> None:
        super().finalize_metrics()
        if RANK not in {-1, 0}:
            return
        plot_pose_metrics(self.metrics, Path(self.save_dir) / "pose_metrics.png")
        if self.current_epoch > 0:
            ordered = [self._sample_records[path] for path in self._sample_paths if path in self._sample_records]
            plot_validation_samples(
                ordered,
                Path(self.save_dir) / "samples" / f"val_samples_e{self.current_epoch}.png",
            )

    def print_results(self) -> None:
        standard_keys = super(UkneePoseMetrics, self.metrics).keys
        standard_values = super(UkneePoseMetrics, self.metrics).mean_results()
        LOGGER.info(
            ("%22s" + "%11i" * 2 + "%11.3g" * len(standard_keys))
            % ("all", self.seen, self.metrics.nt_per_class.sum(), *standard_values)
        )
        if self._medical_records:
            medical = self.metrics.medical
            LOGGER.info(
                "Medical: MRE %.3f mm | HD95 %.3f mm | PCK@0.2/0.4/0.8mm %.3f/%.3f/%.3f",
                medical["metrics/MRE"] * PIXEL_SPACING_MM,
                medical["metrics/HD95"] * PIXEL_SPACING_MM,
                medical["metrics/PCK2"],
                medical["metrics/PCK4"],
                medical["metrics/PCK8"],
            )


class KneePoseTrainer(FlatPoseTrainerMixin, PoseTrainer):
    """Upstream pose trainer with flat outputs and medical validation."""

    def get_validator(self):
        super().get_validator()  # configure the exact Pose26/V1/V9 loss names
        return KneePoseValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )


__all__ = [
    "FlatPoseTrainerMixin",
    "KneePoseTrainer",
    "KneePoseValidator",
    "PIXEL_SPACING_MM",
    "RESULT_COLUMNS",
    "UkneePoseMetrics",
]
