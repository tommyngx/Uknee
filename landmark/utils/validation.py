"""YOLO pose validation extended with knee-specific medical metrics."""

from __future__ import annotations

from copy import copy
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from ultralytics.models.yolo.pose import PoseTrainer, PoseValidator
from ultralytics.utils import LOGGER, RANK, ops
from ultralytics.utils.metrics import PoseMetrics

from landmark.data.schema import LANDMARK_PATH_RANGES, REGION_KEYPOINT_COUNTS, REGION_NAMES


MEDICAL_KEYS = (
    "metrics/MRE",
    "metrics/median_error",
    "metrics/p95_error",
    "metrics/PCK2",
    "metrics/PCK4",
    "metrics/PCK8",
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
        # Preserve upstream checkpoint selection and early-stopping behaviour.
        return super().fitness


class KneePoseValidator(PoseValidator):
    """Add MRE/PCK/failure/topology metrics without changing upstream mAP."""

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.metrics = UkneePoseMetrics()
        self._medical_records: list[dict[str, Any]] = []

    def init_metrics(self, model: torch.nn.Module) -> None:
        super().init_metrics(model)
        self._medical_records.clear()

    def update_metrics(self, preds: list[dict[str, torch.Tensor]], batch: dict[str, Any]) -> None:
        super().update_metrics(preds, batch)
        for image_index, prediction in enumerate(preds):
            self._record_medical(image_index, prediction, batch)

    def _record_medical(self, image_index: int, prediction: dict[str, torch.Tensor], batch: dict[str, Any]) -> None:
        pbatch = self._prepare_batch(image_index, batch)
        classes = pbatch["cls"].long()
        diagonal = float(np.hypot(*pbatch["ori_shape"]))
        all_errors: list[np.ndarray] = []
        region_errors: dict[int, np.ndarray] = {}
        pred_canonical = np.full((sum(REGION_KEYPOINT_COUNTS), 2), np.nan, dtype=np.float32)
        gt_canonical = np.full_like(pred_canonical, np.nan)
        offset = 0

        for class_id, count in enumerate(REGION_KEYPOINT_COUNTS):
            target_rows = torch.nonzero(classes == class_id, as_tuple=False).flatten()
            target = pbatch["keypoints"][target_rows[0], :count].clone() if target_rows.numel() else None
            pred_rows = torch.nonzero(prediction["cls"].long() == class_id, as_tuple=False).flatten()
            selected = pred_rows[prediction["conf"][pred_rows].argmax()] if pred_rows.numel() else None
            predicted = prediction["keypoints"][selected, :count].clone() if selected is not None else None
            if target is None:
                offset += count
                continue

            target = ops.scale_coords(
                pbatch["imgsz"], target, pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"]
            )
            visible = target[:, 2] > 0 if target.shape[-1] == 3 else torch.ones(count, dtype=torch.bool, device=target.device)
            gt_xy = target[:, :2]
            gt_canonical[offset : offset + count] = gt_xy.detach().cpu().numpy()
            if predicted is None:
                errors = np.full(int(visible.sum()), diagonal, dtype=np.float32)
            else:
                predicted = ops.scale_coords(
                    pbatch["imgsz"], predicted, pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"]
                )
                pred_xy = predicted[:, :2]
                pred_canonical[offset : offset + count] = pred_xy.detach().cpu().numpy()
                errors = torch.linalg.vector_norm(pred_xy[visible] - gt_xy[visible], dim=-1).detach().cpu().numpy()
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
            valid = np.isfinite(pred_path).all(1) & np.isfinite(gt_path).all(1)
            if valid.sum() < 2:
                topology_failures.append(True)
                continue
            pred_delta = np.diff(pred_path[valid], axis=0)
            gt_delta = np.diff(gt_path[valid], axis=0)
            agreement = np.sum(pred_delta * gt_delta, axis=1) >= 0
            order_values.extend(agreement.tolist())
            topology_failures.append(not bool(np.all(agreement)))
        self._medical_records.append(
            {
                "errors": errors,
                "regions": region_errors,
                "failed": bool(errors.mean() > 8.0 or np.any(errors >= diagonal)),
                "order": order_values,
                "topology": topology_failures,
            }
        )

    def gather_stats(self) -> None:
        super().gather_stats()
        if RANK == 0:
            gathered: list[Any] = [None] * dist.get_world_size()
            dist.gather_object(self._medical_records, gathered, dst=0)
            self._medical_records = [record for rank_records in gathered for record in (rank_records or [])]
        elif RANK > 0:
            dist.gather_object(self._medical_records, None, dst=0)
            self._medical_records.clear()

    def get_stats(self) -> dict[str, Any]:
        stats = super().get_stats()
        records = self._medical_records
        if records:
            errors = np.concatenate([record["errors"] for record in records])
            values = {
                "metrics/MRE": float(errors.mean()),
                "metrics/median_error": float(np.median(errors)),
                "metrics/p95_error": float(np.percentile(errors, 95)),
                "metrics/PCK2": float((errors <= 2).mean()),
                "metrics/PCK4": float((errors <= 4).mean()),
                "metrics/PCK8": float((errors <= 8).mean()),
                "metrics/failure_rate": float(np.mean([record["failed"] for record in records])),
                "metrics/order_accuracy": float(
                    np.mean([value for record in records for value in record["order"]])
                ),
                "metrics/topology_failure_rate": float(
                    np.mean([value for record in records for value in record["topology"]])
                ),
            }
            for class_id, name in enumerate(REGION_NAMES):
                region = [record["regions"][class_id] for record in records if class_id in record["regions"]]
                values[f"metrics/MRE_{name}"] = float(np.concatenate(region).mean()) if region else float("nan")
            self.metrics.medical.update(values)
            stats.update(values)
        return stats

    def print_results(self) -> None:
        # PoseMetrics' per-class rows do not have meaningful per-class medical
        # aggregate columns, so print the standard summary followed by one
        # explicit medical line.
        standard_keys = super(UkneePoseMetrics, self.metrics).keys
        standard_values = super(UkneePoseMetrics, self.metrics).mean_results()
        LOGGER.info(
            ("%22s" + "%11i" * 2 + "%11.3g" * len(standard_keys))
            % ("all", self.seen, self.metrics.nt_per_class.sum(), *standard_values)
        )
        if self._medical_records:
            LOGGER.info(
                "Medical: MRE %.3f px | median %.3f | p95 %.3f | PCK2/4/8 %.3f/%.3f/%.3f",
                self.metrics.medical["metrics/MRE"],
                self.metrics.medical["metrics/median_error"],
                self.metrics.medical["metrics/p95_error"],
                self.metrics.medical["metrics/PCK2"],
                self.metrics.medical["metrics/PCK4"],
                self.metrics.medical["metrics/PCK8"],
            )


class KneePoseTrainer(PoseTrainer):
    """Upstream pose trainer using :class:`KneePoseValidator`."""

    def get_validator(self):
        # Let upstream select the exact loss names for Pose26, V1 and V9.
        super().get_validator()
        return KneePoseValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )


__all__ = ["KneePoseTrainer", "KneePoseValidator", "UkneePoseMetrics"]
