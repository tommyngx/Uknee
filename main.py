import os
import argparse
import time
import traceback
import re
from pathlib import Path

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
gpu_parser.add_argument('--gpu', type=str, default="7", help='gpu')
temp_args, _ = gpu_parser.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = temp_args.gpu
print(f"Set CUDA_VISIBLE_DEVICES to {os.environ['CUDA_VISIBLE_DEVICES']}")



import random
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(cpu_num)
torch.multiprocessing.set_sharing_strategy('file_system')

import torch.optim as optim
import csv
device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
from models import build_model
import utils.losses as losses
from utils.metrics_medpy import get_metrics
from utils.util import AverageMeter
from utils.training_logs import (
    EpochLogWriter,
    plot_training_dashboard,
    save_training_args,
    setup_logger,
)
from utils.segmentation_reporting import SegmentationEvaluator, plot_segmentation_metrics
from dataloader.dataloader import getDataloader,getZeroShotDataloader
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent
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


def _str2bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, received: {value}")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="U_Net", help='model')
    parser.add_argument('--base_dir', type=str, default="./data/busi", help='data base dir')
    parser.add_argument('--dataset_name', type=str, default="busi", help='dataset_name')
    parser.add_argument('--train_file_dir', type=str, default="train.txt", help='train_file_dir')
    parser.add_argument('--val_file_dir', type=str, default="val.txt", help='val_file_dir')
    parser.add_argument('--base_lr', type=float, default=0.01,
                        help='segmentation network learning rate')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch_size per gpu')
    parser.add_argument('--workers', type=int, default=4, help='DataLoader worker processes')
    parser.add_argument('--gpu', type=str, default="7", help='gpu')
    parser.add_argument('--max_epochs', type=int, default=2, help='epoch')
    parser.add_argument('--seed', type=int, default=2006, help='Reproducibility seed')
    parser.add_argument('--img_size', type=int, default=256, help='img_size')
    parser.add_argument('--num_classes', type=int, default=1, help='img_size')
    parser.add_argument('--input_channel', type=int, default=3, help='img_size')
    parser.add_argument(
        '--aug_strategy',
        type=str,
        default="auto",
        help='2D augmentation strategy: auto, none, basic, standard, strong, xray',
    )
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--pretrained_path', type=str, default="", help='Path to custom pretrained weights/checkpoint (.pth)')
    parser.add_argument('--exp_name', type=str, default="", help='Optional run name; defaults to <model>_<dataset>')
    parser.add_argument('--output_dir', type=str, default="", help='Run root; defaults to <repo>/runs/segmentation')
    parser.add_argument('--pixel_spacing_mm', type=float, default=0.10, help='In-plane pixel spacing used by HD95/ASSD')
    parser.add_argument('--zero_shot_base_dir', type=str, default="", help='zero_base_dir')
    parser.add_argument('--zero_shot_dataset_name', type=str, default="", help='zero_shot_dataset_name')
    parser.add_argument('--do_deeps', type=_str2bool, nargs='?', const=True, default=False, help='Use deep supervision')
    parser.add_argument('--model_id', type=int, default=0, help='model_id')
    parser.add_argument('--just_for_test', type=_str2bool, nargs='?', const=True, default=False, help='Only run validation')
    parser.add_argument('--just_for_zero_shot', type=_str2bool, nargs='?', const=True, default=False, help='Only run zero-shot validation')
    args = parser.parse_args()
    try:
        from dataloader.dataset_mesko import infer_mesko_num_classes, is_mesko_dataset

        if is_mesko_dataset(args.base_dir, args.dataset_name):
            inferred_num_classes = infer_mesko_num_classes(args.base_dir)
            if inferred_num_classes and inferred_num_classes > 1 and int(args.num_classes) != inferred_num_classes:
                print(f"Auto-updating num_classes from {args.num_classes} to {inferred_num_classes} based on MESKO dataset metadata")
                args.num_classes = inferred_num_classes
    except Exception as exc:
        print(f"Could not auto-configure MESKO num_classes: {exc}")
    seed_torch(args.seed)
    return args


args = parse_arguments()


def _requested_gpu_count(gpu_arg):
    return len([gpu_id.strip() for gpu_id in str(gpu_arg).split(',') if gpu_id.strip()])


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


def _load_model_state_dict(model, state_dict, logger=None):
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
    except RuntimeError:
        pass

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
    if args.model == "RWKV_UNet" and int(args.img_size) > 256:
        raise ValueError(
            "RWKV_UNet in this repo only supports img_size <= 256. "
            f"Received img_size={args.img_size}. The custom WKV CUDA kernel is compiled with T_MAX=1024, "
            "so 512x512 inputs overflow the stage attention token limit."
        )
    if args.model == "RWKV_UNetV2" and int(args.img_size) > 1024:
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


def _save_checkpoint(path, args, model, optimizer, epoch, best_dice, metrics=None):
    checkpoint = {
        'epoch': epoch,
        'state_dict': _unwrap_model(model).state_dict(),
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'best_dice': best_dice,
        'metrics': convert_to_numpy(metrics or {}),
        'config': vars(args),
    }
    torch.save(checkpoint, path)



def load_model(args, model_best_or_final="best"):
    exp_save_dir= args.exp_save_dir
    model = build_model(args, input_channel=args.input_channel, num_classes=args.num_classes).to(device)
    if model_best_or_final == "best":
        candidate_paths = [os.path.join(exp_save_dir, 'best.pt')]
    else:
        candidate_paths = [os.path.join(exp_save_dir, 'last.pt')]

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

    _load_model_state_dict(model, state_dict)

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
    run_name = _safe_run_component(args.exp_name) if args.exp_name else default_name
    output_root = Path(args.output_dir).expanduser().resolve() if args.output_dir else REPO_ROOT / "runs" / "segmentation"
    exp_save_path = output_root if output_root.name == run_name else output_root / run_name
    exp_save_path.mkdir(parents=True, exist_ok=True)
    samples_path = exp_save_path / "samples"
    samples_path.mkdir(exist_ok=True)

    args.exp_name = run_name
    args.exp_save_dir = str(exp_save_path)
    args.samples_dir = str(samples_path)

    if not args.resume and not args.just_for_test:
        for filename in ("best.pt", "last.pt", "results.csv", "dashboard_segmentation.png", "segmentation_metrics.png"):
            artifact = exp_save_path / filename
            if artifact.is_file():
                artifact.unlink()
        for artifact in samples_path.glob("val_samples_e*.png"):
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
    plot_segmentation_metrics(snapshot, Path(args.exp_save_dir) / "segmentation_metrics.png")
    evaluator.save_samples(Path(args.samples_dir) / "val_samples_e0.png", epoch=0)
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
        checkpoint_path = run_dir / "last.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Cannot resume: '{checkpoint_path}' does not exist")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        _load_model_state_dict(model, checkpoint["state_dict"], logger=logger)
        if checkpoint.get("optimizer") is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0))
        best_dice = float(checkpoint.get("best_dice", best_dice))
        history_rows = [row for row in history_rows if int(row["epoch"]) <= start_epoch]

    logger.info("model=%s parameters=%d optimizer=%s criterion=%s output=%s",
                args.model, sum(p.numel() for p in model.parameters() if p.requires_grad),
                optimizer_name, criterion_name, run_dir)
    logger.info("train_batches=%d val_batches=%d seed=%d pixel_spacing=%.4f mm/px",
                len(trainloader), len(valloader), args.seed, args.pixel_spacing_mm)

    max_iterations = max(len(trainloader) * args.max_epochs, 1)
    iteration = start_epoch * len(trainloader)
    fixed_indices = SegmentationEvaluator.fixed_sample_indices(len(valloader.dataset), seed=2006)
    training_started = time.time()
    best_epoch = max((int(row["epoch"]) for row in history_rows if row["val/dice"] == best_dice), default=0)

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

        _save_checkpoint(run_dir / "last.pt", args, model, optimizer, epoch_id, max(best_dice, snapshot.dice), epoch_row)
        if snapshot.dice > best_dice:
            best_dice = snapshot.dice
            best_epoch = epoch_id
            _save_checkpoint(run_dir / "best.pt", args, model, optimizer, epoch_id, best_dice, epoch_row)

        evaluator.save_samples(run_dir / "samples" / f"val_samples_e{epoch_id}.png", epoch_id)
        plot_segmentation_metrics(snapshot, run_dir / "segmentation_metrics.png")
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
            filename="dashboard_segmentation.png",
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

    logger.info("Training complete: best Dice %.4f at epoch %d", best_dice, best_epoch)
    return {"best_dice": best_dice, "best_epoch": best_epoch, "run_dir": str(run_dir)}









if __name__ == "__main__":

    
    print(f"\n=== Testing model: {args.model} ===")
    _validate_runtime_config(args)

    exp_save_dir, log_dir, history_writer, logger, model = init_dir(args)
    if args.just_for_test:
        model, model_path = load_model(args, model_best_or_final="best")
        print(f"Test-only mode: loaded {model_path}")
        validate(args, logger, model)
        if args.zero_shot_dataset_name:
            zero_shot(args, logger, model)
        raise SystemExit(0)
    try:
        train(args, exp_save_dir, log_dir, history_writer, logger, model)
        if args.zero_shot_dataset_name:
            zero_shot(args, logger, model)
        print(f"Model {args.model} training finished successfully")
    except Exception as e:
        logger.exception("Training failed with an exception.")
        traceback.print_exc()
        print(f"Model {args.model} failed: {str(e)}")
        raise SystemExit(1)
    
