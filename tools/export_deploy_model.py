import argparse
import json
import random
from pathlib import Path
import sys
from types import SimpleNamespace
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import build_model


VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class SegmentationInferenceWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)
        if isinstance(output, (list, tuple)):
            output = output[-1]
        elif isinstance(output, dict):
            output = output["out"] if "out" in output else next(iter(output.values()))
        return output


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a segmentation checkpoint to a deployable artifact and test it on random samples."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint file path or experiment directory containing best_models/checkpoint_top1.pth.",
    )
    parser.add_argument("--data_dir", type=str, required=True, help="Dataset root with images/ and masks/")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save exported model and preview")
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument(
        "--export_format",
        type=str,
        default="auto",
        choices=["auto", "torchscript", "bundle"],
        help="Deployment artifact format. 'auto' uses TorchScript when possible and falls back to bundle.",
    )
    parser.add_argument("--num_samples", type=int, default=3, help="Number of random samples for preview")
    parser.add_argument("--seed", type=int, default=41, help="Random seed for sample selection")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binary threshold for sigmoid output")
    return parser.parse_args()


def resolve_checkpoint_path(checkpoint_arg: str):
    checkpoint_path = Path(checkpoint_arg).expanduser().resolve()
    if checkpoint_path.is_file():
        return checkpoint_path

    if checkpoint_path.is_dir():
        candidates = [
            checkpoint_path / "best_models" / "checkpoint_top1.pth",
            checkpoint_path / "checkpoint_last.pth",
            checkpoint_path / "checkpoint_final.pth",
            checkpoint_path / "checkpoint_best.pth",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate

    raise FileNotFoundError(f"Could not resolve checkpoint from: {checkpoint_arg}")


def load_checkpoint_config(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config_dict = checkpoint.get("config")

    if config_dict is None:
        config_path = checkpoint_path.parent.parent / "configs" / "config.json"
        if config_path.is_file():
            with config_path.open("r", encoding="utf-8") as file:
                config_dict = json.load(file)
        else:
            raise KeyError(
                f"Checkpoint '{checkpoint_path}' does not contain config and no config.json was found nearby."
            )

    return checkpoint, SimpleNamespace(**config_dict)


def namespace_to_dict(config):
    if isinstance(config, dict):
        return dict(config)
    return vars(config).copy()


def resolve_device(device_arg: str, model_name: str):
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)

    if "RWKV" in model_name and device.type != "cuda":
        raise SystemExit(
            f"Model '{model_name}' uses CUDA-only custom ops in this repo. Please export on a CUDA device."
        )
    return device


def discover_mask_dir(data_dir: Path):
    mask_zero_dir = data_dir / "masks" / "0"
    masks_dir = data_dir / "masks"
    if mask_zero_dir.is_dir():
        return mask_zero_dir
    if masks_dir.is_dir():
        return masks_dir
    raise FileNotFoundError(
        f"Mask directory not found. Expected either {data_dir / 'masks' / '0'} or {data_dir / 'masks'}."
    )


def collect_files(directory: Path):
    file_map = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in VALID_EXTENSIONS:
            continue
        file_map.setdefault(path.stem, path)
    return file_map


def choose_random_pairs(data_dir: Path, num_samples: int, seed: int):
    images_dir = data_dir / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    mask_dir = discover_mask_dir(data_dir)
    image_map = collect_files(images_dir)
    mask_map = collect_files(mask_dir)
    paired_stems = sorted(set(image_map) & set(mask_map))
    if not paired_stems:
        raise RuntimeError("No paired image/mask samples found.")

    rng = random.Random(seed)
    sample_count = min(num_samples, len(paired_stems))
    chosen_stems = rng.sample(paired_stems, sample_count)
    return [(stem, image_map[stem], mask_map[stem]) for stem in chosen_stems]


def load_image_for_model(image_path: Path, img_size: int, input_channel: int):
    with Image.open(image_path) as image:
        display_image = image.convert("RGB").resize((img_size, img_size), Image.BILINEAR)
        if input_channel == 1:
            model_image = display_image.convert("L")
            image_array = np.array(model_image, dtype=np.float32)[..., None]
        else:
            image_array = np.array(display_image, dtype=np.float32)

    image_array = image_array / 255.0
    tensor = torch.from_numpy(image_array.transpose(2, 0, 1)).unsqueeze(0)
    return tensor, np.array(display_image)


def load_mask_for_preview(mask_path: Path, img_size: int):
    with Image.open(mask_path) as mask:
        mask = mask.convert("L").resize((img_size, img_size), Image.NEAREST)
        mask_array = np.array(mask, dtype=np.uint8)
    mask_array = (mask_array > 0).astype(np.uint8) * 255
    return mask_array


def render_prediction(logits: torch.Tensor, threshold: float):
    if logits.ndim != 4:
        raise ValueError(f"Expected model output with shape [B, C, H, W], got {tuple(logits.shape)}")

    if logits.shape[1] == 1:
        probabilities = torch.sigmoid(logits)
        prediction = (probabilities >= threshold).to(torch.uint8) * 255
        return prediction[0, 0].cpu().numpy()

    prediction = torch.argmax(logits, dim=1).to(torch.uint8)
    max_label = int(max(logits.shape[1] - 1, 1))
    prediction = (prediction * (255 // max_label)).cpu().numpy()
    return prediction[0]


def save_preview(output_path: Path, preview_rows):
    num_rows = len(preview_rows)
    fig, axes = plt.subplots(num_rows, 3, figsize=(12, 4 * num_rows))
    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_index, row in enumerate(preview_rows):
        axes[row_index, 0].imshow(row["image"])
        axes[row_index, 0].set_title(f"{row['stem']} | Image")
        axes[row_index, 1].imshow(row["gt"], cmap="gray", vmin=0, vmax=255)
        axes[row_index, 1].set_title(f"{row['stem']} | GT")
        axes[row_index, 2].imshow(row["pred"], cmap="gray", vmin=0, vmax=255)
        axes[row_index, 2].set_title(f"{row['stem']} | Predict")

        for col in range(3):
            axes[row_index, col].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def export_torchscript_model(model: nn.Module, dummy_input: torch.Tensor, output_path: Path):
    with torch.no_grad():
        traced_model = torch.jit.trace(model, dummy_input, strict=False)
        traced_model = torch.jit.freeze(traced_model.eval())
        traced_model.save(str(output_path))


def export_bundle_model(output_path: Path, config_dict, state_dict, checkpoint_path: Path):
    bundle = {
        "format": "deploy_bundle",
        "config": config_dict,
        "state_dict": state_dict,
        "source_checkpoint": str(checkpoint_path),
    }
    torch.save(bundle, output_path)


def load_bundle_model(bundle_path: Path, device: torch.device):
    bundle = torch.load(bundle_path, map_location=device, weights_only=False)
    config = SimpleNamespace(**bundle["config"])
    model = build_model(
        config,
        input_channel=int(config.input_channel),
        num_classes=int(config.num_classes),
    ).to(device)
    model.load_state_dict(bundle["state_dict"], strict=False)
    model.eval()
    return SegmentationInferenceWrapper(model).to(device).eval(), bundle


def should_prefer_bundle(export_format: str, model_name: str):
    if export_format == "bundle":
        return True
    if export_format == "torchscript":
        return False
    return "RWKV" in model_name


def main():
    args = parse_args()
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    data_dir = Path(args.data_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint, config = load_checkpoint_config(checkpoint_path)
    device = resolve_device(args.device, config.model)

    model = build_model(
        config,
        input_channel=int(config.input_channel),
        num_classes=int(config.num_classes),
    ).to(device)

    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    config_dict = namespace_to_dict(config)

    wrapper = SegmentationInferenceWrapper(model).to(device).eval()
    dummy_input = torch.randn(
        1,
        int(config.input_channel),
        int(config.img_size),
        int(config.img_size),
        device=device,
    )

    export_format = args.export_format
    export_notes = []

    if should_prefer_bundle(export_format, config.model):
        deploy_model_path = output_dir / "model_deploy_bundle.pth"
        export_bundle_model(
            output_path=deploy_model_path,
            config_dict=config_dict,
            state_dict=state_dict,
            checkpoint_path=checkpoint_path,
        )
        converted_model, bundle_info = load_bundle_model(deploy_model_path, device)
        actual_export_format = "bundle"
        if export_format == "auto" and "RWKV" in config.model:
            export_notes.append(
                "Auto mode selected deploy bundle because this model uses custom RWKV/WKV ops that do not export cleanly to TorchScript."
            )
    else:
        deploy_model_path = output_dir / "model_deploy.ts"
        try:
            export_torchscript_model(wrapper, dummy_input, deploy_model_path)
            converted_model = torch.jit.load(str(deploy_model_path), map_location=device).eval()
            actual_export_format = "torchscript"
        except Exception as exc:
            if export_format == "torchscript":
                raise
            warnings.warn(
                f"TorchScript export failed for model '{config.model}'. Falling back to deploy bundle. "
                f"Original error: {exc}",
                RuntimeWarning,
            )
            deploy_model_path = output_dir / "model_deploy_bundle.pth"
            export_bundle_model(
                output_path=deploy_model_path,
                config_dict=config_dict,
                state_dict=state_dict,
                checkpoint_path=checkpoint_path,
            )
            converted_model, bundle_info = load_bundle_model(deploy_model_path, device)
            actual_export_format = "bundle"
            export_notes.append(f"TorchScript export failed and auto mode fell back to bundle: {exc}")

    sample_pairs = choose_random_pairs(data_dir, args.num_samples, args.seed)

    preview_rows = []
    with torch.no_grad():
        for stem, image_path, mask_path in sample_pairs:
            input_tensor, display_image = load_image_for_model(
                image_path=image_path,
                img_size=int(config.img_size),
                input_channel=int(config.input_channel),
            )
            gt_mask = load_mask_for_preview(mask_path, img_size=int(config.img_size))
            logits = converted_model(input_tensor.to(device))
            if not torch.isfinite(logits).all():
                raise RuntimeError(
                    f"Converted model produced non-finite output for sample '{stem}'."
                )
            pred_mask = render_prediction(logits, threshold=args.threshold)
            preview_rows.append(
                {
                    "stem": stem,
                    "image": display_image,
                    "gt": gt_mask,
                    "pred": pred_mask,
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                }
            )

    preview_path = output_dir / f"random_preview_{len(preview_rows)}.png"
    save_preview(preview_path, preview_rows)

    report = {
        "checkpoint_path": str(checkpoint_path),
        "exported_model_path": str(deploy_model_path),
        "export_format": actual_export_format,
        "preview_path": str(preview_path),
        "model_name": config.model,
        "img_size": int(config.img_size),
        "input_channel": int(config.input_channel),
        "num_classes": int(config.num_classes),
        "device": str(device),
        "threshold": float(args.threshold),
        "samples": [
            {
                "stem": row["stem"],
                "image_path": row["image_path"],
                "mask_path": row["mask_path"],
            }
            for row in preview_rows
        ],
    }
    if export_notes:
        report["export_notes"] = export_notes
    if "RWKV" in config.model:
        report["runtime_note"] = (
            "RWKV models in this project rely on the registered custom WKV operator. "
            "The deploy bundle keeps config + weights together, but inference still needs this repo/runtime."
        )

    report_path = output_dir / "deployment_report.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("=" * 72)
    print("Deploy Export Summary")
    print("=" * 72)
    print(f"Checkpoint           : {checkpoint_path}")
    print(f"Exported model       : {deploy_model_path}")
    print(f"Export format        : {actual_export_format}")
    print(f"Preview image        : {preview_path}")
    print(f"Report               : {report_path}")
    print(f"Model                : {config.model}")
    print(f"Device               : {device}")
    print(f"Random samples       : {len(preview_rows)}")
    if export_notes:
        for note in export_notes:
            print(f"Note                 : {note}")
    for row in preview_rows:
        print(f"- {row['stem']}: {row['image_path']} | {row['mask_path']}")


if __name__ == "__main__":
    main()
