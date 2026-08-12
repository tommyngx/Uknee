import os
import argparse
import time
import traceback
import re
import shutil
import sys
from pathlib import Path

# Direct ``python segment/main.py`` needs the repository root before shared imports.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from uknee_cli import parse_gpu_ids

cpu_num = 1
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/uknee-matplotlib')
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)

gpu_parser = argparse.ArgumentParser(add_help=False)
gpu_parser.add_argument('--gpu', type=parse_gpu_ids, default=[0], help='GPU list, e.g. [0,1]')
temp_args, _ = gpu_parser.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, temp_args.gpu)) if temp_args.gpu != [-1] else ""
print(f"Set CUDA_VISIBLE_DEVICES to {os.environ['CUDA_VISIBLE_DEVICES']}")



import random
import warnings
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings(
    "ignore",
    message=r"`torch\.cuda\.amp\.autocast\(args\.\.\.\)` is deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"nll_loss2d_forward_out_cuda_template does not have a deterministic implementation.*",
    category=UserWarning,
)

torch.set_num_threads(cpu_num)
torch.multiprocessing.set_sharing_strategy('file_system')

import torch.optim as optim
import csv
device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
# Allow this file to be run directly as well as with ``python -m``.

from segment.models import build_model
import segment.utils.losses as losses
from segment.utils.metrics_medpy import get_metrics
from segment.utils.util import AverageMeter
from segment.utils.training_logs import (
    EpochLogWriter,
    load_summary_yaml,
    model_paper_profile,
    plot_training_dashboard,
    save_summary_yaml,
    save_training_args,
    setup_logger,
)
from segment.utils.segment_reporting import SegmentationEvaluator, plot_segmentation_metrics
from segment.utils.onnx_export import (
    SUPPORTED_AUTO_EXPORT_MODELS,
    export_segment_onnx,
    onnx_filename,
    segment_preprocess_schema,
)
from segment.dataloader.dataloader import getDataloader, getZeroShotDataloader
from segment.cli import parse_segment_args
from segment.utils.preprocessing import resolve_target_hw
import torch.nn.functional as F
RESULT_COLUMNS = [
    "epoch",
    "train/loss",
    "val/loss",
    "val/dice",
    "val/iou",
    "val/hd95",
    "val/assd",
    "val/sens",
    "val/prec",
]


def _should_export_pending_best(epoch, total_epochs, best_epoch, last_exported_best_epoch, interval=10):
    """Export at epoch 1, each interval, and final epoch only when best.pt changed."""
    epoch = int(epoch)
    interval = max(int(interval), 1)
    scheduled = epoch == 1 or epoch % interval == 0 or epoch == int(total_epochs)
    return scheduled and int(best_epoch) > int(last_exported_best_epoch)

def convert_to_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    elif isinstance(data, dict):
        return {key: convert_to_numpy(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_numpy(item) for item in data]
    else:
        return data


def seed_torch(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(True,warn_only=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def parse_arguments(argv=None):
    args = parse_segment_args(argv)
    try:
        from segment.dataloader.dataset_mesko import infer_mesko_num_classes, is_mesko_dataset

        if is_mesko_dataset(args.base_dir, args.dataset_name):
            inferred_num_classes = infer_mesko_num_classes(args.base_dir)
            if inferred_num_classes and inferred_num_classes > 1 and int(args.num_classes) != inferred_num_classes:
                print(f"Auto-updating num_classes from {args.num_classes} to {inferred_num_classes} based on MESKO dataset metadata")
                args.num_classes = inferred_num_classes
    except Exception as exc:
        print(f"Could not auto-configure MESKO num_classes: {exc}")
    seed_torch(args.seed)
    return args


def _requested_gpu_count(gpu_arg):
    return len(parse_gpu_ids(gpu_arg))


def _unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def _wrap_data_parallel_if_needed(model, args, logger=None):
    if not torch.cuda.is_available():
        return model

    visible_gpu_count = torch.cuda.device_count()
    requested_gpu_count = _requested_gpu_count(args.gpu)
    if visible_gpu_count <= 1 or requested_gpu_count <= 1:
        return model

    device_ids = list(range(visible_gpu_count))
    if logger is not None:
        logger.info(
            "Using DataParallel on visible CUDA devices %s from --gpu=%s.",
            device_ids,
            args.gpu,
        )
    else:
        print(f"Using DataParallel on visible CUDA devices {device_ids} from --gpu={args.gpu}")
    return nn.DataParallel(model, device_ids=device_ids)


def _load_model_state_dict(model, state_dict, logger=None, *, strict=False):
    target = _unwrap_model(model)
    model_state = target.state_dict()

    if any(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }

    try:
        target.load_state_dict(state_dict, strict=True)
        return
    except RuntimeError as exc:
        if strict:
            raise RuntimeError(
                "Checkpoint architecture does not match the requested model. "
                "Legacy compact RWKV_UNetV6 checkpoints were renamed to RWKV_UNetV5; "
                "resume those runs with --model RWKV_UNetV5."
            ) from exc

    matched_state_dict = {}
    mismatched_keys = []

    for k, v in state_dict.items():
        if k in model_state:
            if v.shape == model_state[k].shape:
                matched_state_dict[k] = v
            else:
                mismatched_keys.append(f"{k} (checkpoint shape {tuple(v.shape)} vs model shape {tuple(model_state[k].shape)})")

    target.load_state_dict(matched_state_dict, strict=False)

    msg = (
        f"Flexible checkpoint loading complete: loaded {len(matched_state_dict)}/{len(model_state)} parameters. "
        f"Skipped {len(mismatched_keys)} mismatched layers."
    )
    if logger is not None:
        logger.info(msg)
        if mismatched_keys:
            logger.info(f"Mismatched layers skipped: {mismatched_keys}")
    else:
        print(msg)
        if mismatched_keys:
            print(f"Mismatched layers skipped: {mismatched_keys}")


def _validate_runtime_config(args):
    height, width = resolve_target_hw(args.img_size)
    longest_side = max(height, width)
    if not Path(args.base_dir).is_dir():
        raise FileNotFoundError(
            f"Dataset folder does not exist: {args.base_dir}. "
            "Use an absolute path or /name for <project>/data/name."
        )
    if args.model == "RWKV_UNet" and longest_side > 256:
        raise ValueError(
            "RWKV_UNet in this repo only supports img_size <= 256. "
            f"Received img_size={args.img_size}. The custom WKV CUDA kernel is compiled with T_MAX=1024, "
            "so 512x512 inputs overflow the stage attention token limit."
        )
    if args.model == "RWKV_UNetV2" and longest_side > 1024:
        raise ValueError(
            "RWKV_UNetV2 uses strip-wise RWKV mixing and supports img_size <= 1024. "
            f"Received img_size={args.img_size}. The custom WKV CUDA kernel is compiled with T_MAX=1024."
        )



def deep_supervision_loss(outputs, label_batch, loss_metric,weights=None):
    num=len(outputs)

    total_loss = 0.0

    for i, output in enumerate(outputs):
        spatial_dims = output.dim() - 2
        target_spatial_size = label_batch.shape[-spatial_dims:]
        if output.shape[-spatial_dims:] != target_spatial_size:
            interpolate_mode = {1: "linear", 2: "bilinear", 3: "trilinear"}.get(spatial_dims, "nearest")
            align_corners = True if interpolate_mode in {"linear", "bilinear", "trilinear"} else None
            output = F.interpolate(
                output,
                size=target_spatial_size,
                mode=interpolate_mode,
                align_corners=align_corners,
            )
        loss = loss_metric(output, label_batch)
        total_loss += loss

    return total_loss/ num


def _build_criterion(args):
    if int(args.num_classes) > 1:
        return losses.__dict__['DiceCELoss'](n_classes=args.num_classes).to(device), "DiceCELoss"
    return losses.__dict__['BCEDiceLoss']().to(device), "BCEDiceLoss"


def _as_float(value):
    value = convert_to_numpy(value)
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return float("nan")
        return float(value.reshape(-1)[0])
    if value is None:
        return float("nan")
    return float(value)


def _tensor_stats(name, tensor):
    if not torch.is_tensor(tensor):
        return f"{name}: non-tensor value={tensor}"

    detached = tensor.detach()
    shape = tuple(detached.shape)
    finite_mask = torch.isfinite(detached)
    finite_ratio = float(finite_mask.float().mean().item()) if detached.numel() else 1.0

    if finite_mask.any():
        finite_values = detached[finite_mask]
        min_value = float(finite_values.min().item())
        max_value = float(finite_values.max().item())
        mean_value = float(finite_values.mean().item())
        std_value = float(finite_values.std().item()) if finite_values.numel() > 1 else 0.0
    else:
        min_value = float("nan")
        max_value = float("nan")
        mean_value = float("nan")
        std_value = float("nan")

    return (
        f"{name}: shape={shape}, dtype={detached.dtype}, finite_ratio={finite_ratio:.4f}, "
        f"min={min_value:.6f}, max={max_value:.6f}, mean={mean_value:.6f}, std={std_value:.6f}"
    )


def _raise_non_finite_error(logger, epoch, batch_idx, loss_value, volume_batch, label_batch, outputs):
    logger.error(
        "Non-finite value detected at epoch [%d], batch [%d]. loss=%s",
        epoch,
        batch_idx,
        loss_value,
    )
    logger.error(_tensor_stats("input", volume_batch))
    logger.error(_tensor_stats("label", label_batch))
    logger.error(_tensor_stats("output", outputs))
    raise RuntimeError(
        "Non-finite training value detected. Review the console log for input, label, and output statistics."
    )


def _build_optimizer(args, model, logger):
    if "RWKV" in args.model:
        effective_lr = args.base_lr if args.base_lr <= 1e-3 else 1e-4
        if effective_lr != args.base_lr:
            logger.warning(
                "Model %s is prone to instability with lr=%s. Switching to AdamW with lr=%s.",
                args.model,
                args.base_lr,
                effective_lr,
            )
        else:
            logger.info("Using AdamW optimizer for %s with lr=%s.", args.model, effective_lr)
        optimizer = optim.AdamW(model.parameters(), lr=effective_lr, weight_decay=0.0001)
        return optimizer, effective_lr, "AdamW"

    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
    return optimizer, args.base_lr, "SGD"


def _save_checkpoint(path, args, model, optimizer, epoch, best_dice, metrics=None, *, include_optimizer=True):
    """Save a resumable last.pt or a compact inference-only best.pt."""
    checkpoint = {
        'checkpoint_type': 'resume' if include_optimizer else 'inference_best',
        'epoch': epoch,
        'state_dict': _unwrap_model(model).state_dict(),
        'best_dice': best_dice,
        'metrics': convert_to_numpy(metrics or {}),
        'config': vars(args),
        'preprocess': segment_preprocess_schema(args),
    }
    if include_optimizer:
        checkpoint['optimizer'] = optimizer.state_dict() if optimizer is not None else None
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        torch.save(checkpoint, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _compact_existing_best_checkpoint(path):
    """Remove legacy optimizer state from best.pt without changing its model weights."""
    path = Path(path)
    if not path.is_file():
        return False
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "optimizer" not in checkpoint:
        return False
    checkpoint.pop("optimizer", None)
    checkpoint["checkpoint_type"] = "inference_best"
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(checkpoint, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return True



def load_model(args, model_best_or_final="best"):
    exp_save_dir= args.exp_save_dir
    model = build_model(args, input_channel=args.input_channel, num_classes=args.num_classes).to(device)
    if model_best_or_final == "best":
        candidate_paths = [
            os.path.join(exp_save_dir, 'weights', 'best.pt'),
            os.path.join(exp_save_dir, 'best.pt'),  # legacy flat layout
        ]
    else:
        candidate_paths = [
            os.path.join(exp_save_dir, 'weights', 'last.pt'),
            os.path.join(exp_save_dir, 'last.pt'),  # legacy flat layout
        ]

    model_path = next((path for path in candidate_paths if os.path.exists(path)), None)
    if model_path is None:
        raise FileNotFoundError(
            f"Could not find a '{model_best_or_final}' checkpoint under '{exp_save_dir}'. "
            f"Checked: {candidate_paths}"
        )

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    _load_model_state_dict(model, state_dict, strict=True)

    model.to(device)
    model = _wrap_data_parallel_if_needed(model, args)

    return model, model_path

def zero_shot(args,logger,model=None):
    valloader = getZeroShotDataloader(args)
    if model is None:
        model,model_path = load_model(args)

    logger.info("train file dir:{} val file dir:{}".format(args.train_file_dir, args.val_file_dir))
    criterion, _ = _build_criterion(args)

    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'val_loss': AverageMeter(),
                  'val_iou': AverageMeter(),
                  'SE': AverageMeter(),
                  'PC': AverageMeter(),
                  'F1': AverageMeter(),
                  'ACC': AverageMeter()
                  }
    model.eval()

    with torch.no_grad():
        for i_batch, sampled_batch in tqdm(enumerate(valloader), total=len(valloader), desc="Zero-shot Validation"):
            input, target = sampled_batch['image'], sampled_batch['label']
            input = input.to(device)
            target = target.to(device)
            output = model(input)
            output = output[-1] if args.do_deeps else output
            loss = criterion(output, target)
            
            iou, _, SE, PC, F1, _, ACC = get_metrics(output, target)
            avg_meters['val_loss'].update(loss.item(), input.size(0))
            avg_meters['val_iou'].update(iou, input.size(0))
            avg_meters['SE'].update(SE, input.size(0))
            avg_meters['PC'].update(PC, input.size(0))
            avg_meters['F1'].update(F1, input.size(0))
            avg_meters['ACC'].update(ACC, input.size(0))
    logger.info(f"zero shot on {args.zero_shot_dataset_name}")
    logger.info('val_loss %.4f - val_iou %.4f - val_SE %.4f - val_PC %.4f - val_F1 %.4f - val_ACC %.4f'
        % (avg_meters['val_loss'].avg, avg_meters['val_iou'].avg, avg_meters['SE'].avg,
            avg_meters['PC'].avg, avg_meters['F1'].avg, avg_meters['ACC'].avg))

    
    zero_shot_result = {"zeroshot_loss":avg_meters['val_loss'].avg, "zeroshot_iou":avg_meters['val_iou'].avg, "zeroshot_SE":avg_meters['SE'].avg,
            "zeroshot_PC":avg_meters['PC'].avg, "zeroshot_F1":avg_meters['F1'].avg, "zeroshot_ACC":avg_meters['ACC'].avg}
    zero_shot_result = convert_to_numpy(zero_shot_result)
    return zero_shot_result


def _safe_run_component(value):
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip())
    return value.strip("._-") or "run"


def init_dir(args):
    default_name = f"{_safe_run_component(args.model)}_{_safe_run_component(args.dataset_name)}"
    run_name = _safe_run_component(args.name) if args.name else default_name
    output_root = Path(args.output_dir).expanduser().resolve() if args.output_dir else Path(args.project) / "runs"
    output_root.mkdir(parents=True, exist_ok=True)
    exp_save_path = output_root / run_name
    exp_save_path.mkdir(parents=True, exist_ok=bool(args.exist_ok))
    weights_path = exp_save_path / "weights"
    weights_path.mkdir(exist_ok=True)
    samples_path = exp_save_path / "samples"
    samples_path.mkdir(exist_ok=True)

    args.name = args.exp_name = run_name
    args.exp_save_dir = str(exp_save_path)
    args.weights_dir = str(weights_path)
    args.samples_dir = str(samples_path)

    if args.resume:
        for filename in ("best.pt", "last.pt"):
            legacy_path = exp_save_path / filename
            destination = weights_path / filename
            if legacy_path.is_file() and not destination.exists():
                shutil.copy2(legacy_path, destination)

    if not args.resume and not args.just_for_test:
        for filename in (
            "best.pt", "last.pt", "results.csv", "summary.yaml",
            "dashboard_segmentation.png", "segmentation_metrics.png",
            "segment_dashboard.png", "segment_metrics.png", onnx_filename(args.model),
        ):
            artifact = exp_save_path / filename
            if artifact.is_file():
                artifact.unlink()
        for filename in ("best.pt", "last.pt", onnx_filename(args.model)):
            (weights_path / filename).unlink(missing_ok=True)
        for pattern in ("val_samples_e*.png", "segment_sample_e*.png", "segment_sample_eval.png"):
            for artifact in samples_path.glob(pattern):
                artifact.unlink()

    args_path = save_training_args(exp_save_path, vars(args), filename="args.yaml")
    print(f"Config saved to {args_path}")

    logger = setup_logger(
        log_file=None,
        logger_name=f"uknee.main.{args.model}.{args.dataset_name}.{run_name}",
    )
    history_writer = EpochLogWriter(
        exp_save_path,
        file_stem="results",
        fieldnames=RESULT_COLUMNS,
        write_auxiliary=False,
    )
    model = build_model(config=args,input_channel=args.input_channel, num_classes=args.num_classes).to(device)
    model = _wrap_data_parallel_if_needed(model, args, logger)

    return str(exp_save_path), str(exp_save_path), history_writer, logger, model


def _dataset_class_names(dataset):
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    class_info = getattr(dataset, "class_info", None)
    if class_info:
        return [item.get("name", f"Region {index}") for index, item in enumerate(class_info)]
    return None


def _export_best_segment_onnx(args, model, weights_dir, run_dir, class_names, logger, *, load_best=False):
    """Atomically refresh the deployment ONNX from the current best checkpoint."""
    if not args.auto_export_onnx:
        return {"status": "disabled", "path": None}
    if args.model not in SUPPORTED_AUTO_EXPORT_MODELS:
        record = {
            "status": "not_supported",
            "path": None,
            "supported_models": sorted(SUPPORTED_AUTO_EXPORT_MODELS),
        }
        logger.info(
            "Automatic ONNX export is currently scoped to %s; skipping model %s.",
            record["supported_models"],
            args.model,
        )
        return record

    best_checkpoint_path = weights_dir / "best.pt"
    if not best_checkpoint_path.is_file():
        raise FileNotFoundError(f"Cannot export ONNX because best checkpoint is missing: {best_checkpoint_path}")
    export_model = model
    if load_best:
        checkpoint = torch.load(best_checkpoint_path, map_location="cpu", weights_only=False)
        export_model = build_model(
            args,
            input_channel=args.input_channel,
            num_classes=args.num_classes,
        ).cpu()
        _load_model_state_dict(export_model, checkpoint["state_dict"], logger=logger, strict=True)

    onnx_path = weights_dir / onnx_filename(args.model)
    temporary = onnx_path.with_name(f".{onnx_path.stem}.tmp.onnx")
    temporary.unlink(missing_ok=True)
    logger.info("Updating ONNX from best checkpoint: %s", onnx_path)
    try:
        record = export_segment_onnx(
            export_model,
            args,
            temporary,
            class_names=class_names,
            validate=True,
        )
        temporary.replace(onnx_path)
    except Exception as exc:
        logger.exception(
            "ONNX export failed for best checkpoint; training will continue and retry after the next best epoch."
        )
        return {
            "status": "failed",
            "path": onnx_path.relative_to(run_dir).as_posix() if onnx_path.is_file() else None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        temporary.unlink(missing_ok=True)
    record["path"] = onnx_path.relative_to(run_dir).as_posix()
    logger.info(
        "ONNX export ready: %s (max_abs=%.6g mean_abs=%.6g p99_abs=%.6g mask_agreement=%.4f%%)",
        onnx_path,
        record["parity"]["max_abs_diff"],
        record["parity"]["mean_abs_diff"],
        record["parity"]["p99_abs_diff"],
        100.0 * record["parity"]["postprocess_agreement"],
    )
    return record


def _validate_epoch(args, model, valloader, criterion, sample_indices=()):
    val_loss = AverageMeter()
    evaluator = SegmentationEvaluator(
        num_classes=args.num_classes,
        pixel_spacing_mm=args.pixel_spacing_mm,
        class_names=_dataset_class_names(valloader.dataset),
        sample_indices=sample_indices,
        seed=args.seed,
    )
    model.eval()
    dataset_offset = 0
    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(valloader):
            input, target = sampled_batch['image'], sampled_batch['label']
            input = input.to(device)
            target = target.to(device)
            output = model(input)
            output = output[-1] if args.do_deeps else output
            loss = criterion(output, target)
            if not torch.isfinite(output).all() or not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite validation result in batch {i_batch}")
            val_loss.update(loss.item(), input.size(0))
            evaluator.update(output, target, images=input, start_index=dataset_offset)
            dataset_offset += input.size(0)
    return _as_float(val_loss.avg), evaluator, evaluator.snapshot()


def validate(args,logger,model):
    _, valloader = getDataloader(args)
    criterion, _ = _build_criterion(args)
    indices = SegmentationEvaluator.fixed_sample_indices(len(valloader.dataset), seed=2006)
    val_loss, evaluator, snapshot = _validate_epoch(args, model, valloader, criterion, indices)
    plot_segmentation_metrics(snapshot, Path(args.exp_save_dir) / "segment_metrics.png")
    evaluator.save_samples(Path(args.samples_dir) / "segment_sample_eval.png", epoch=0)
    metrics = {
        "val/loss": val_loss,
        "val/dice": snapshot.dice,
        "val/iou": snapshot.iou,
        "val/hd95": snapshot.hd95,
        "val/assd": snapshot.assd,
        "val/sens": snapshot.sensitivity,
        "val/prec": snapshot.precision,
    }
    logger.info("Validation metrics: %s", metrics)
    return metrics


def _read_history(results_path):
    if not Path(results_path).is_file():
        return []
    rows = []
    with Path(results_path).open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            rows.append({key: int(value) if key == "epoch" else float(value) for key, value in row.items()})
    return rows


def train(args, exp_save_dir, log_dir, history_writer, logger, model):
    trainloader, valloader = getDataloader(args)
    optimizer, base_lr, optimizer_name = _build_optimizer(args, model, logger)
    criterion, criterion_name = _build_criterion(args)
    run_dir = Path(exp_save_dir)
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    args.effective_lr = base_lr
    args.optimizer = optimizer_name
    args.criterion = criterion_name
    save_training_args(run_dir, vars(args), filename="args.yaml")
    results_path = run_dir / "results.csv"
    history_rows = _read_history(results_path) if args.resume else []
    start_epoch = 0
    best_dice = max((row["val/dice"] for row in history_rows), default=-1.0)

    if args.pretrained_path:
        checkpoint = torch.load(args.pretrained_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint)) if isinstance(checkpoint, dict) else checkpoint
        _load_model_state_dict(model, state_dict, logger=logger)
    elif args.resume:
        checkpoint_path = next(
            (path for path in (weights_dir / "last.pt", run_dir / "last.pt") if path.is_file()),
            None,
        )
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"Cannot resume: neither '{weights_dir / 'last.pt'}' nor legacy '{run_dir / 'last.pt'}' exists"
            )
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        _load_model_state_dict(model, checkpoint["state_dict"], logger=logger, strict=True)
        if checkpoint.get("optimizer") is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0))
        best_dice = float(checkpoint.get("best_dice", best_dice))
        history_rows = [row for row in history_rows if int(row["epoch"]) <= start_epoch]
        if _compact_existing_best_checkpoint(weights_dir / "best.pt"):
            logger.info("Compacted legacy best.pt by removing optimizer state; last.pt remains resumable.")

    logger.info("model=%s parameters=%d optimizer=%s criterion=%s output=%s",
                args.model, sum(p.numel() for p in model.parameters() if p.requires_grad),
                optimizer_name, criterion_name, run_dir)
    logger.info("train_batches=%d val_batches=%d seed=%d pixel_spacing=%.4f mm/px",
                len(trainloader), len(valloader), args.seed, args.pixel_spacing_mm)

    max_iterations = max(len(trainloader) * args.max_epochs, 1)
    iteration = start_epoch * len(trainloader)
    fixed_indices = SegmentationEvaluator.fixed_sample_indices(len(valloader.dataset), seed=2006)
    previous_summary = load_summary_yaml(run_dir) if args.resume else {}
    previous_duration = float(previous_summary.get("training", {}).get("duration_seconds", 0.0) or 0.0)
    training_started = time.time()
    best_epoch = max((int(row["epoch"]) for row in history_rows if row["val/dice"] == best_dice), default=0)
    onnx_record = None
    onnx_path = weights_dir / onnx_filename(args.model)
    best_path = weights_dir / "best.pt"
    last_exported_best_epoch = (
        best_epoch
        if onnx_path.is_file() and best_path.is_file() and onnx_path.stat().st_mtime_ns >= best_path.stat().st_mtime_ns
        else 0
    )

    for epoch_num in range(start_epoch, args.max_epochs):
        epoch_id = epoch_num + 1
        model.train()
        train_loss = AverageMeter()
        progress = tqdm(trainloader, desc=f"Epoch [{epoch_id}/{args.max_epochs}] Train", dynamic_ncols=True)
        for sampled_batch in progress:
            learning_rate = base_lr * max(1.0 - iteration / max_iterations, 0.0) ** 0.9
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            images = sampled_batch["image"].to(device)
            targets = sampled_batch["label"].to(device)
            if args.do_deeps:
                outputs = model(images)
                loss = deep_supervision_loss(outputs, targets, criterion)
                final_output = outputs[-1]
            else:
                final_output = model(images)
                loss = criterion(final_output, targets)
            if not torch.isfinite(final_output).all() or not torch.isfinite(loss):
                _raise_non_finite_error(logger, epoch_id, iteration, loss.detach(), images, targets, final_output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iteration += 1
            train_loss.update(loss.item(), images.size(0))
            progress.set_postfix(loss=f"{train_loss.avg:.4f}", lr=f"{learning_rate:.2e}")

        val_loss, evaluator, snapshot = _validate_epoch(
            args, model, valloader, criterion, sample_indices=fixed_indices
        )
        epoch_row = {
            "epoch": epoch_id,
            "train/loss": _as_float(train_loss.avg),
            "val/loss": val_loss,
            "val/dice": snapshot.dice,
            "val/iou": snapshot.iou,
            "val/hd95": snapshot.hd95,
            "val/assd": snapshot.assd,
            "val/sens": snapshot.sensitivity,
            "val/prec": snapshot.precision,
        }
        history_rows.append(epoch_row)
        history_writer.append(epoch_row)

        _save_checkpoint(
            weights_dir / "last.pt", args, model, optimizer, epoch_id,
            max(best_dice, snapshot.dice), epoch_row, include_optimizer=True,
        )
        if snapshot.dice > best_dice:
            best_dice = snapshot.dice
            best_epoch = epoch_id
            _save_checkpoint(
                weights_dir / "best.pt", args, model, optimizer, epoch_id,
                best_dice, epoch_row, include_optimizer=False,
            )
        if _should_export_pending_best(
            epoch_id,
            args.max_epochs,
            best_epoch,
            last_exported_best_epoch,
            args.onnx_export_interval,
        ):
            logger.info(
                "ONNX export checkpoint reached at epoch %d; exporting best.pt from epoch %d.",
                epoch_id,
                best_epoch,
            )
            onnx_record = _export_best_segment_onnx(
                args,
                model,
                weights_dir,
                run_dir,
                _dataset_class_names(valloader.dataset),
                logger,
                load_best=True,
            )
            if onnx_record.get("status") == "ready":
                last_exported_best_epoch = best_epoch

        evaluator.save_samples(run_dir / "samples" / f"segment_sample_e{epoch_id}.png", epoch_id)
        plot_segmentation_metrics(snapshot, run_dir / "segment_metrics.png")
        plot_training_dashboard(
            log_dir=run_dir,
            history_rows=history_rows,
            loss_keys=[("train/loss", "Training Loss"), ("val/loss", "Validation Loss")],
            metric_keys=[
                ("val/dice", "Val Dice"), ("val/iou", "Val IoU"),
                ("val/hd95", "HD95"), ("val/assd", "ASSD"),
                ("val/sens", "Sensitivity"), ("val/prec", "Precision"),
            ],
            ranking_key="val/dice",
            maximize=True,
            top_k=3,
            filename="segment_dashboard.png",
            model_name=args.model,
            title=f"{args.model} | {args.dataset_name}",
            elapsed_seconds=time.time() - training_started,
        )
        logger.info(
            "Epoch %d/%d | train_loss=%.6f val_loss=%.6f dice=%.4f iou=%.4f "
            "HD95=%.3fmm ASSD=%.3fmm sens=%.4f prec=%.4f",
            epoch_id, args.max_epochs, epoch_row["train/loss"], val_loss, snapshot.dice,
            snapshot.iou, snapshot.hd95, snapshot.assd, snapshot.sensitivity, snapshot.precision,
        )

    elapsed_seconds = previous_duration + time.time() - training_started
    best_row = next((row for row in history_rows if int(row["epoch"]) == best_epoch), {})
    final_row = history_rows[-1] if history_rows else {}
    preprocess_schema = segment_preprocess_schema(args)
    if best_epoch > last_exported_best_epoch:
        onnx_record = _export_best_segment_onnx(
            args,
            model,
            weights_dir,
            run_dir,
            _dataset_class_names(valloader.dataset),
            logger,
            load_best=True,
        )
        if onnx_record.get("status") == "ready":
            last_exported_best_epoch = best_epoch
    elif onnx_record is None:
        previous_onnx_record = previous_summary.get("deployment", {}).get("onnx", {})
        onnx_record = previous_onnx_record if onnx_path.is_file() and previous_onnx_record else (
            {"status": "ready", "path": onnx_path.relative_to(run_dir).as_posix()}
            if onnx_path.is_file()
            else {"status": "not_generated", "path": None}
        )
    summary = {
        "schema_version": 2,
        "task": "segmentation",
        "model": {
            "name": args.model,
            **model_paper_profile(
                model,
                (1, int(args.input_channel), *resolve_target_hw(args.img_size)),
            ),
        },
        "dataset": {
            "name": args.dataset_name,
            "path": str(Path(args.base_dir).resolve()),
            "train_manifest": args.train_file_dir,
            "validation_manifest": args.val_file_dir,
            "classes": int(args.num_classes),
            "pixel_spacing_mm": float(args.pixel_spacing_mm),
        },
        "preprocessing": preprocess_schema,
        "training": {
            "epochs_requested": int(args.max_epochs),
            "epochs_completed": int(final_row.get("epoch", start_epoch)),
            "batch_size": int(args.batch_size),
            "seed": int(args.seed),
            "gpu_ids": list(args.gpu_ids),
            "optimizer": optimizer_name,
            "criterion": criterion_name,
            "initial_learning_rate": float(base_lr),
            "duration_seconds": round(elapsed_seconds, 3),
            "duration_hours": round(elapsed_seconds / 3600.0, 6),
            "seconds_per_epoch": round(elapsed_seconds / max(len(history_rows), 1), 3),
            "device": str(device),
            "torch_version": str(torch.__version__),
        },
        "performance": {
            "selection_metric": "val/dice",
            "selection_mode": "max",
            "best_epoch": int(best_epoch),
            "best": best_row,
            "final": final_row,
            "distance_unit": "mm",
        },
        "deployment": {
            "auto_export_onnx": bool(args.auto_export_onnx),
            "onnx_export_interval": int(args.onnx_export_interval),
            "last_exported_best_epoch": int(last_exported_best_epoch),
            "onnx": onnx_record,
        },
        "artifacts": {
            "best_checkpoint": "weights/best.pt",
            "best_checkpoint_type": "inference_best_without_optimizer",
            "last_checkpoint": "weights/last.pt",
            "last_checkpoint_type": "resumable_with_optimizer",
            "metrics": "results.csv",
            "dashboard": "segment_dashboard.png",
            "metric_report": "segment_metrics.png",
            "samples": "samples/segment_sample_e{epoch}.png",
            "samples_per_epoch": min(4, len(valloader.dataset)),
            "sample_seed": 2006,
            "sample_indices": fixed_indices,
            "sample_display_height": 512,
            "sample_display_width": "preserve_aspect_ratio",
            "sample_output_width": 800,
            "onnx_model": onnx_record.get("path"),
        },
    }
    save_summary_yaml(run_dir, summary)
    logger.info("Training complete: best Dice %.4f at epoch %d", best_dice, best_epoch)
    return {"best_dice": best_dice, "best_epoch": best_epoch, "run_dir": str(run_dir)}









def main(argv=None):
    args = parse_arguments(argv)
    if os.environ.get("UKNEE_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}:
        print(f"\n=== Testing model: {args.model} ===")
    _validate_runtime_config(args)

    exp_save_dir, log_dir, history_writer, logger, model = init_dir(args)
    if args.just_for_test:
        model, model_path = load_model(args, model_best_or_final="best")
        print(f"Test-only mode: loaded {model_path}")
        validate(args, logger, model)
        if args.zero_shot_dataset_name:
            zero_shot(args, logger, model)
        return {"status": "tested", "run_dir": exp_save_dir}
    try:
        train(args, exp_save_dir, log_dir, history_writer, logger, model)
        if args.zero_shot_dataset_name:
            zero_shot(args, logger, model)
        print(f"Model {args.model} training finished successfully")
    except KeyboardInterrupt:
        logger.warning(
            "Training interrupted by user. The latest completed-epoch checkpoint is preserved; "
            "restart with --resume to continue."
        )
        return {"status": "interrupted", "run_dir": exp_save_dir}
    except Exception as e:
        logger.exception("Training failed with an exception.")
        traceback.print_exc()
        print(f"Model {args.model} failed: {str(e)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
