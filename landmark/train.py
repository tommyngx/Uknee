from __future__ import annotations

import argparse
import json
import logging
import math
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from landmark.config import load_config
from landmark.data import build_dataloaders
from landmark.losses import LandmarkLoss
from landmark.models import build_model
from landmark.utils.checkpoint import save_checkpoint
from landmark.utils.metrics import landmark_metrics
from landmark.utils.plotting import TrainingPlotter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Uknee landmark model")
    parser.add_argument(
        "--config",
        default="landmark/config/adaptive_rwkv.yaml",
        help="Experiment YAML path",
    )
    parser.add_argument("--resume", default="", help="Training checkpoint to resume")
    parser.add_argument("--epochs", type=int, default=None, help="Override YAML epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--device", default=None, help="cuda, mps, cpu, or auto")
    return parser.parse_args()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _phase(model: torch.nn.Module, epoch: int, coarse_epochs: int, teacher_epochs: int) -> str:
    if not hasattr(model, "set_training_phase"):
        return "full"
    if epoch <= coarse_epochs:
        phase = "coarse"
    elif epoch <= coarse_epochs + teacher_epochs:
        phase = "refinement"
    else:
        phase = "full"
    model.set_training_phase(phase)
    return phase


def _teacher_reference(
    batch: dict[str, torch.Tensor],
    noise_pixels: float,
    image_height: int,
    image_width: int,
) -> torch.Tensor:
    scale = batch["landmarks"].new_tensor([image_width - 1, image_height - 1])
    noise = torch.randn_like(batch["landmarks"]) * noise_pixels / scale
    return (batch["landmarks"] + noise).clamp(0, 1)


def _run_train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    phase,
    config,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = defaultdict(float)
    sample_count = 0
    amp_enabled = config.training.amp and device.type == "cuda"
    progress = tqdm(loader, desc=f"train/{phase}", leave=False)
    for batch in progress:
        batch = _move_batch(batch, device)
        reference = None
        if phase == "refinement" and hasattr(model, "set_training_phase"):
            reference = _teacher_reference(
                batch,
                config.training.teacher_noise_pixels,
                config.data.image_height,
                config.data.image_width,
            )
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(batch["image"], reference_landmarks=reference)
            losses = criterion(outputs, batch, phase=phase)
        scaler.scale(losses["loss"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            config.training.gradient_clip,
        )
        scaler.step(optimizer)
        scaler.update()
        batch_size = batch["image"].shape[0]
        sample_count += batch_size
        for key, value in losses.items():
            totals[key] += float(value.detach()) * batch_size
        progress.set_postfix(loss=f"{float(losses['loss'].detach()):.4f}")
    return {key: value / max(sample_count, 1) for key, value in totals.items()}


@torch.no_grad()
def _run_validation(model, loader, criterion, device, config):
    model.eval()
    totals: dict[str, float] = defaultdict(float)
    predictions, targets, visibility = [], [], []
    sample_count = 0
    last_batch = last_outputs = None
    for batch in tqdm(loader, desc="validate", leave=False):
        batch = _move_batch(batch, device)
        outputs = model(batch["image"])
        losses = criterion(outputs, batch, phase="full")
        batch_size = batch["image"].shape[0]
        sample_count += batch_size
        for key, value in losses.items():
            totals[key] += float(value.detach()) * batch_size
        predictions.append(outputs["final_landmarks"].detach().cpu())
        targets.append(batch["landmarks"].detach().cpu())
        visibility.append(batch["landmark_visibility"].detach().cpu())
        last_batch, last_outputs = batch, outputs
    metrics = landmark_metrics(
        torch.cat(predictions),
        torch.cat(targets),
        torch.cat(visibility),
        config.data.image_height,
        config.data.image_width,
    )
    losses = {key: value / max(sample_count, 1) for key, value in totals.items()}
    return losses, metrics, last_batch, last_outputs


def _scheduler(optimizer, epochs: int, warmup_epochs: int):
    def schedule(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.device is not None:
        config.training.device = args.device
    if args.resume:
        config.training.resume = args.resume
    _seed_everything(config.training.seed)
    device = _device(config.training.device)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = (
        Path(config.training.output_dir)
        / config.training.experiment_name
        / timestamp
    )
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    with (run_dir / "config_resolved.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config.to_dict(), stream, sort_keys=False)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(run_dir / "logs" / "train.log"),
        ],
    )
    logging.info("Run directory: %s", run_dir)
    logging.info("Device: %s", device)

    train_loader, val_loader = build_dataloaders(
        config.data, config.training.batch_size
    )
    assert train_loader is not None
    model = build_model(config).to(device)
    criterion = LandmarkLoss(config.loss).to(device)
    # Capture all decoder parameters before curriculum temporarily freezes modules.
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = _scheduler(
        optimizer, config.training.epochs, config.training.warmup_epochs
    )
    scaler = GradScaler(device.type, enabled=config.training.amp and device.type == "cuda")
    start_epoch, best_mre = 1, float("inf")

    if config.training.resume:
        checkpoint = torch.load(
            config.training.resume, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scheduler"):
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_mre = float(checkpoint.get("best_metric", best_mre))

    plotter = TrainingPlotter(run_dir)
    for epoch in range(start_epoch, config.training.epochs + 1):
        phase = _phase(
            model,
            epoch,
            config.training.coarse_only_epochs,
            config.training.teacher_forcing_epochs,
        )
        train_losses = _run_train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            phase,
            config,
        )
        val_losses, metrics, sample_batch, sample_outputs = _run_validation(
            model, val_loader, criterion, device, config
        )
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_losses["loss"],
            "val_loss": val_losses["loss"],
            "coarse_loss": val_losses["coarse_loss"],
            "coordinate_loss": val_losses["coordinate_loss"],
            "heatmap_loss": val_losses["heatmap_loss"],
            "val_mre": metrics["mre"],
            "val_pck4": metrics["pck4"],
            "val_pck8": metrics["pck8"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        plotter.update(row)
        if sample_batch is not None:
            plotter.overlay(
                sample_batch["image"].detach().cpu(),
                sample_outputs["final_landmarks"].detach().cpu(),
                sample_batch["landmarks"].detach().cpu(),
                sample_batch["landmark_visibility"].detach().cpu(),
                epoch,
                config.training.plot_samples,
            )
        logging.info(
            "epoch=%d phase=%s train=%.5f val=%.5f mre=%.3f pck4=%.4f",
            epoch,
            phase,
            train_losses["loss"],
            val_losses["loss"],
            metrics["mre"],
            metrics["pck4"],
        )
        if metrics["mre"] < best_mre:
            best_mre = metrics["mre"]
            save_checkpoint(
                run_dir / "checkpoints" / "best.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_mre,
                config.to_dict(),
            )
        if epoch % config.training.save_every == 0 or epoch == config.training.epochs:
            save_checkpoint(
                run_dir / "checkpoints" / f"epoch_{epoch:04d}.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_mre,
                config.to_dict(),
            )
    (run_dir / "result.json").write_text(
        json.dumps({"best_mre": best_mre}, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
