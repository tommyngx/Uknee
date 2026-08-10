"""Public Uknee pose API backed by the vendored Ultralytics snapshot."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import torch
import yaml

from ultralytics import YOLO

from landmark.data.prepare import PreparedDataset, prepare_dataset
from landmark.utils.results import KneePoseResult, adapt_yolo_result


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG = PACKAGE_ROOT / "cfg" / "default.yaml"
SUPPORTED_EXPORT_FORMATS = {"onnx", "torchscript"}
SINGLE_IMAGE_MODELS = {"yolo26-pose-v1", "yolo26-pose-v9", "hrnet-w32-pose", "vitpose-s-pose"}
LEGACY_MODEL_MARKERS = ("adaptive_rwkv", "adaptive_detr", "kneepv1", "kneepv2", "kneept640v0")


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected a YAML mapping in {path}")
    return data


def _is_heatmap_config(path: Path) -> bool:
    return path.suffix.lower() in {".yaml", ".yml"} and _load_yaml(path).get("landmark_family") == "heatmap"


class KneePose:
    """Train, validate, predict and export one of Uknee's five public models.

    YOLO checkpoints remain native Ultralytics checkpoints. Predictions preserve
    the standard ``Results`` object through :class:`KneePoseResult` and add the
    canonical 129-landmark representation used by Uknee.
    """

    def __init__(self, model: str | Path, *, verbose: bool = False) -> None:
        self.model_path = Path(model).expanduser()
        if self.model_path.suffix.lower() in {".yaml", ".yml", ".pt"} and not self.model_path.is_file():
            raise FileNotFoundError(f"Model file does not exist: {self.model_path}")
        self.model_path = self.model_path.resolve()
        lowered = self.model_path.name.lower()
        if any(marker in lowered for marker in LEGACY_MODEL_MARKERS):
            raise ValueError(
                f"Legacy landmark checkpoint/config '{self.model_path.name}' is not compatible with the "
                "vendored YOLO26/heatmap runtime. Retrain or export a plain state-dict with the legacy code."
            )
        self.family = "heatmap" if _is_heatmap_config(self.model_path) else "yolo"
        if self.family == "heatmap":
            from landmark.heatmap.engine import HeatmapPose

            self.backend = HeatmapPose(self.model_path, verbose=verbose)
        else:
            try:
                self.backend = YOLO(str(self.model_path), task="pose", verbose=verbose)
            except Exception as error:
                if self.model_path.suffix.lower() == ".pt":
                    raise ValueError(
                        f"Checkpoint '{self.model_path}' is not compatible with the vendored Uknee pose runtime. "
                        "Expected an Ultralytics 8.4.87 YOLO26 pose checkpoint or a new HRNet/ViTPose checkpoint."
                    ) from error
                raise
            if type(self.backend.model).__module__.startswith("landmark.heatmap"):
                from landmark.heatmap.engine import HeatmapPose

                self.family = "heatmap"
                self.backend = HeatmapPose(self.model_path, verbose=verbose)
        if self.family == "yolo" and isinstance(self.backend.model, torch.nn.Module):
            # PoseTrainer normally attaches these attributes. YAML inference
            # must also work before the first training run.
            head = self.backend.model.model[-1]
            self.backend.model.kpt_shape = list(head.kpt_shape)
            self.backend.model.nc = int(head.nc)
            self.backend.model.names = {0: "femur", 1: "tibia", 2: "fibula", 3: "patella"}

    @property
    def model(self):
        """Underlying PyTorch model."""
        return self.backend.model

    def _prepare_data(self, data: str | Path) -> PreparedDataset:
        return prepare_dataset(data)

    def train(self, *, data: str | Path, **kwargs: Any):
        """Train with leakage-safe validated data and upstream trainer defaults."""
        prepared = self._prepare_data(data)
        args = _load_yaml(DEFAULT_CFG)
        args.update(kwargs)
        args["data"] = str(prepared.yaml_path)
        head_name = type(self.model.model[-1]).__name__ if self.family == "yolo" else ""
        if (
            self.model_path.stem in SINGLE_IMAGE_MODELS
            or head_name in {"OA26HeatmapPose", "OA26RegionRefinePose"}
            or self.family == "heatmap"
        ):
            args.update(mosaic=0.0, mixup=0.0, cutmix=0.0)

        def save_resolved_dataset(trainer) -> None:
            from ultralytics.utils import RANK

            if RANK not in {-1, 0}:
                return
            destination = Path(trainer.save_dir) / "dataset_resolved.yaml"
            if not destination.exists():
                shutil.copy2(prepared.yaml_path, destination)

        if hasattr(self.backend, "add_callback"):
            self.backend.add_callback("on_train_start", save_resolved_dataset)
            self.backend.add_callback("on_model_save", _save_best_mre)
        if self.family == "yolo":
            from landmark.utils.validation import KneePoseTrainer

            return self.backend.train(trainer=KneePoseTrainer, **args)
        return self.backend.train(**args)

    def val(self, *, data: str | Path, **kwargs: Any):
        """Run box/pose validation plus Uknee medical landmark metrics."""
        prepared = self._prepare_data(data)
        if self.family == "yolo":
            from landmark.utils.validation import KneePoseValidator

            return self.backend.val(validator=KneePoseValidator, data=str(prepared.yaml_path), **kwargs)
        return self.backend.val(data=str(prepared.yaml_path), **kwargs)

    def predict(self, source: Any = None, *, stream: bool = False, **kwargs: Any):
        """Return YOLO-compatible results enriched with canonical 129 points."""
        results = self.backend.predict(source=source, stream=stream, **kwargs)
        if stream:
            return (adapt_yolo_result(result) for result in results)
        return [adapt_yolo_result(result) for result in results]

    __call__ = predict

    def export(self, *, format: str = "onnx", **kwargs: Any):
        """Export the deployment graph; only ONNX and TorchScript are public API."""
        normalized = format.lower()
        if normalized not in SUPPORTED_EXPORT_FORMATS:
            raise ValueError(
                f"Uknee exposes only {sorted(SUPPORTED_EXPORT_FORMATS)}, got format={format!r}. "
                "Other exporters remain internal to the vendored backend."
            )
        from ultralytics.engine.exporter import Exporter

        from landmark.utils.exporting import KneePoseExportWrapper

        wrapper = KneePoseExportWrapper(self.model, self.family, confidence=float(kwargs.pop("conf", 0.25)))
        # PyTorch's eval fast-path emits aten::_native_multi_head_attention,
        # which has no ONNX symbolic. Disabling only that fused fast-path
        # decomposes V9/ViTPose attention into equivalent exportable ops; it
        # does not alter ROIAlign or ordinary train/inference execution.
        uses_attention = self.family == "heatmap" or type(self.model.model[-1]).__name__ == "OA26RegionRefinePose"
        mha_fastpath = torch.backends.mha.get_fastpath_enabled()
        if normalized == "onnx" and uses_attention:
            torch.backends.mha.set_fastpath_enabled(False)
        try:
            return Exporter(
                overrides={"format": normalized, **kwargs}, _callbacks=self.backend.callbacks
            )(model=wrapper)
        finally:
            if normalized == "onnx" and uses_attention:
                torch.backends.mha.set_fastpath_enabled(mha_fastpath)

    def export_state_dict(self, path: str | Path) -> Path:
        """Write a portable weights-only artifact for paper archives."""
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        model = self.model.module if hasattr(self.model, "module") else self.model
        torch.save(
            {
                "format": "uknee-state-dict-v1",
                "source_model": self.model_path.name,
                "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            },
            destination,
        )
        return destination


__all__ = ["KneePose", "KneePoseResult"]


def _save_best_mre(trainer) -> None:
    """Maintain a medical-error checkpoint independently of upstream fitness."""
    from ultralytics.utils import RANK

    if RANK not in {-1, 0}:
        return
    if not getattr(trainer, "metrics", None):
        return
    current = trainer.metrics.get("metrics/MRE")
    if current is None or not torch.isfinite(torch.tensor(float(current))):
        return
    record = Path(trainer.wdir) / "best_mre.txt"
    if hasattr(trainer, "_uknee_best_mre"):
        best = trainer._uknee_best_mre
    else:
        try:
            best = float(record.read_text(encoding="utf-8").strip())
        except (FileNotFoundError, ValueError):
            best = float("inf")
    if float(current) < best:
        trainer._uknee_best_mre = float(current)
        source = Path(trainer.last)
        if source.exists():
            shutil.copy2(source, Path(trainer.wdir) / "best_mre.pt")
            record.write_text(f"{float(current):.12g}\n", encoding="utf-8")
