from __future__ import annotations

import argparse
import json

import torch
from tqdm import tqdm

from landmark.config import load_config
from landmark.data import build_dataloaders
from landmark.models import build_model
from landmark.utils.metrics import landmark_metrics


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
    with torch.no_grad():
        for batch in tqdm(loader):
            image = batch["image"].to(device)
            outputs = model(image, return_heatmaps=False)
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
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
