from __future__ import annotations

import argparse
import json

import torch
from tqdm import tqdm

from landmark.config import load_config
from landmark.data import build_dataloaders
from landmark.models import build_model
from landmark.utils.metrics import landmark_metrics


def _contour_oracle_statistics(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    image_height: int,
    image_width: int,
) -> tuple[float, int] | None:
    """Return summed visible-point oracle error in pixels and point count."""
    if "contour_candidate_coordinates" not in outputs:
        return None
    scale = outputs["final_landmarks"].new_tensor(
        [max(image_width - 1, 1), max(image_height - 1, 1)]
    )
    offsets = (
        outputs["contour_candidate_coordinates"]
        - batch["landmarks"][:, :, None]
    ) * scale
    nearest = torch.linalg.vector_norm(offsets, dim=-1).min(dim=-1).values
    visible = batch["landmark_visibility"].bool()
    return float(nearest[visible].sum()), int(visible.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a landmark checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device)
    _, loader = build_dataloaders(
        config.data, config.training.batch_size, train=False
    )
    model = build_model(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    predictions, targets, visibility = [], [], []
    oracle_error_sum = 0.0
    oracle_point_count = 0
    with torch.no_grad():
        for batch in tqdm(loader):
            image = batch["image"].to(device)
            device_batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            outputs = model(image, return_heatmaps=False)
            oracle = _contour_oracle_statistics(
                outputs,
                device_batch,
                config.data.image_height,
                config.data.image_width,
            )
            if oracle is not None:
                oracle_error_sum += oracle[0]
                oracle_point_count += oracle[1]
            predictions.append(outputs["final_landmarks"].cpu())
            targets.append(batch["landmarks"])
            visibility.append(batch["landmark_visibility"])
    metrics = landmark_metrics(
        torch.cat(predictions),
        torch.cat(targets),
        torch.cat(visibility),
        config.data.image_height,
        config.data.image_width,
    )
    if oracle_point_count:
        metrics["contour_oracle_px"] = oracle_error_sum / oracle_point_count
    metrics["checkpoint_epoch"] = int(checkpoint.get("epoch", -1))
    metrics["total_parameters"] = sum(
        parameter.numel() for parameter in model.parameters()
    )
    metrics["trainable_parameters"] = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
