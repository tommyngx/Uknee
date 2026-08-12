"""Public Uknee pose API backed by the self-contained landmark runtime."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch
import yaml

from landmark.core.model import YOLO

from landmark.data.prepare import PreparedDataset, prepare_dataset
from landmark.utils.results import KneePoseResult, adapt_yolo_result
from uknee_cli import parse_image_size


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG = PACKAGE_ROOT / "cfg" / "default.yaml"
SUPPORTED_EXPORT_FORMATS = {"onnx", "torchscript"}
SINGLE_IMAGE_MODELS = {
    "yolo26-pose-v1", "yolo26-pose-v9", "hrnet-w32-pose", "hrnet-w48-pose",
    "vitpose-s-pose", "vitpose-b-pose", "rtmo-pose",
}
LEGACY_MODEL_MARKERS = ("adaptive_rwkv", "adaptive_detr", "kneepv1", "kneepv2", "kneept640v0")


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected a YAML mapping in {path}")
    return data


def _model_family(path: Path) -> str:
    if path.suffix.lower() not in {".yaml", ".yml"}:
        return "yolo"
    family = str(_load_yaml(path).get("landmark_family", "yolo"))
    return family if family in {"heatmap", "rtmo"} else "yolo"


class KneePose:
    """Train, validate, predict and export an Uknee landmark model.

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
                "landmark YOLO26/heatmap runtime. Retrain or export a plain state-dict with the legacy code."
            )
        self.family = _model_family(self.model_path)
        if self.family in {"heatmap", "rtmo"}:
            from landmark.models.heatmap_adapter import HeatmapPose

            self.backend = HeatmapPose(self.model_path, verbose=verbose)
        else:
            try:
                self.backend = YOLO(str(self.model_path), task="pose", verbose=verbose)
            except Exception as error:
                if self.model_path.suffix.lower() == ".pt":
                    raise ValueError(
                        f"Checkpoint '{self.model_path}' is not compatible with the landmark pose runtime. "
                        "Expected an Ultralytics 8.4.87 YOLO26 pose checkpoint or a new HRNet/ViTPose/RTMO checkpoint."
                    ) from error
                raise
            if type(self.backend.model).__module__.startswith("landmark.models.heatmap_adapter"):
                from landmark.models.heatmap_adapter import HeatmapPose

                self.family = str(getattr(self.backend.model, "yaml", {}).get("landmark_family", "heatmap"))
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
        data_path = Path(data).expanduser().resolve()
        dataset_config = _load_yaml(data_path)
        dataset_name = str(dataset_config.get("dataset_name", data_path.stem))
        run_name = "_".join(
            re.sub(r"[^a-zA-Z0-9-]+", "-", value).strip("-").lower()
            for value in (self.model_path.stem, dataset_name)
        )
        prepared = self._prepare_data(data)
        args = _load_yaml(DEFAULT_CFG)
        args.update(kwargs)
        args.update(
            # DEFAULT_CFG intentionally keeps model null for generic parsing,
            # but Model.train merges keyword arguments last. Passing that null
            # would overwrite the YAML/checkpoint already loaded by KneePose.
            model=str(self.model_path),
            data=str(prepared.yaml_path),
            dataset_source=str(prepared.source_yaml),
            dataset_root=str(prepared.root),
            plots=False,
            save_period=-1,
            flipud=0.0,
            fliplr=0.0,
            cutmix=0.0,
            copy_paste=0.0,
            erasing=0.0,
            bgr=0.0,
        )
        args["project"] = str(Path(kwargs.get("project") or PACKAGE_ROOT.parent / "runs").expanduser().resolve())
        args["name"] = str(kwargs.get("name") or run_name)
        args["exist_ok"] = bool(kwargs.get("exist_ok", True))
        args["imgsz"] = parse_image_size(args.get("imgsz", 640))
        Path(args["project"]).mkdir(parents=True, exist_ok=True)
        if args["imgsz"][0] != args["imgsz"][1]:
            # Vendored mosaic assumes a square canvas. Rectangular runs use
            # the normal medical-image affine/intensity path plus letterbox.
            args.update(mosaic=0.0, mixup=0.0, cutmix=0.0, multi_scale=0.0, rect=False)
        head_name = type(self.model.model[-1]).__name__ if self.family == "yolo" else ""
        if (
            self.model_path.stem in SINGLE_IMAGE_MODELS
            or head_name in {"OA26HeatmapPose", "OA26RegionRefinePose"}
            or self.family in {"heatmap", "rtmo"}
        ):
            args.update(mosaic=0.0, mixup=0.0, cutmix=0.0)

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
        from landmark.core.exporter import Exporter

        from landmark.utils.exporting import KneePoseExportWrapper

        wrapper = KneePoseExportWrapper(self.model, self.family, confidence=float(kwargs.pop("conf", 0.25)))
        # PyTorch's eval fast-path emits aten::_native_multi_head_attention,
        # which has no ONNX symbolic. Disabling only that fused fast-path
        # decomposes V9/ViTPose attention into equivalent exportable ops; it
        # does not alter ROIAlign or ordinary train/inference execution.
        uses_attention = self.family in {"heatmap", "rtmo"} or type(self.model.model[-1]).__name__ == "OA26RegionRefinePose"
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
