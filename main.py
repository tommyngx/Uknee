import os
import argparse
import shutil
import time
import traceback

cpu_num = 1
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('MPLBACKEND', 'Agg')
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
import matplotlib.pyplot as plt
import json
import logging
import numpy as np
import torch

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
import tempfile
from utils.training_logs import (
    EpochLogWriter,
    plot_training_dashboard,
    save_training_args,
    setup_logger,
)
from dataloader.dataloader import getDataloader,getZeroShotDataloader
import torch.nn.functional as F

def convert_to_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    elif isinstance(data, dict):
        return {key: convert_to_numpy(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_numpy(item) for item in data]
    else:
        return data


def ensure_parent_dir(file_path):
    parent_dir = os.path.dirname(file_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

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
    parser.add_argument('--gpu', type=str, default="7", help='gpu')
    parser.add_argument('--max_epochs', type=int, default=2, help='epoch')
    parser.add_argument('--seed', type=int, default=41, help='seed')
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
    parser.add_argument('--exp_name', type=str, default="default_exp", help='Experiment name')
    parser.add_argument('--output_dir', type=str, default="", help='Base output directory. Run artifacts are saved under {output_dir}/{exp_name}/. Defaults to ./output/{exp_name}/')
    parser.add_argument('--zero_shot_base_dir', type=str, default="", help='zero_base_dir')
    parser.add_argument('--zero_shot_dataset_name', type=str, default="", help='zero_shot_dataset_name')
    parser.add_argument('--do_deeps', type=bool, default=False, help='Use deep supervision')
    parser.add_argument('--model_id', type=int, default=0, help='model_id')
    parser.add_argument('--just_for_test', type=bool, default=0, help='just for test')
    parser.add_argument('--just_for_zero_shot', type=bool, default=0, help='just for test')
    args = parser.parse_args()
    seed_torch(args.seed)
    return args


args = parse_arguments()


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
        if output.shape[1:] != label_batch.shape[1:]:
            output = F.interpolate(output, size=label_batch.shape[1:], mode='bilinear', align_corners=True)
        loss = loss_metric(output, label_batch)
        total_loss += loss

    return total_loss/ num


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
        "Non-finite training value detected. Check logs/training.log for input, label, and output statistics."
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


def _save_checkpoint(path, args, model, optimizer, epoch, best_iou, metrics=None):
    checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'best_iou': best_iou,
        'metrics': convert_to_numpy(metrics or {}),
        'config': vars(args),
    }
    torch.save(checkpoint, path)


def _refresh_topk_aliases(best_model_dir, topk_entries, top_k=3):
    os.makedirs(best_model_dir, exist_ok=True)
    summary = []

    for rank in range(1, top_k + 1):
        alias_path = os.path.join(best_model_dir, f'checkpoint_top{rank}.pth')
        if os.path.exists(alias_path):
            os.remove(alias_path)

    for rank, entry in enumerate(topk_entries[:top_k], start=1):
        alias_path = os.path.join(best_model_dir, f'checkpoint_top{rank}.pth')
        shutil.copy2(entry["path"], alias_path)
        summary.append({
            "rank": rank,
            "epoch": entry["epoch"],
            "value": entry["score"],
            "source_path": entry["path"],
            "alias_path": alias_path,
        })

    summary_path = os.path.join(best_model_dir, 'topk_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as file:
        json.dump(summary, file, indent=4)

    return summary


def _load_topk_entries(best_model_dir):
    summary_path = os.path.join(best_model_dir, 'topk_summary.json')
    if not os.path.exists(summary_path):
        return []

    with open(summary_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    topk_entries = []
    for item in data:
        source_path = item.get("source_path")
        if source_path and os.path.exists(source_path):
            topk_entries.append({
                "epoch": int(item["epoch"]),
                "score": float(item["value"]),
                "path": source_path,
            })
    topk_entries.sort(key=lambda entry: entry["score"], reverse=True)
    return topk_entries


def _maybe_save_topk_checkpoint(best_model_dir, topk_entries, score, epoch, args, model, optimizer, metrics, top_k=3):
    if not np.isfinite(score):
        return topk_entries, _refresh_topk_aliases(best_model_dir, topk_entries, top_k=top_k)

    should_save = len(topk_entries) < top_k or score > topk_entries[-1]["score"]
    if not should_save:
        return topk_entries, _refresh_topk_aliases(best_model_dir, topk_entries, top_k=top_k)

    checkpoint_name = f'epoch_{epoch:03d}_val_iou_{score:.6f}.pth'
    checkpoint_path = os.path.join(best_model_dir, checkpoint_name)
    _save_checkpoint(
        checkpoint_path,
        args=args,
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        best_iou=score,
        metrics=metrics,
    )

    updated_entries = [entry for entry in topk_entries if entry["epoch"] != epoch]
    updated_entries.append({"epoch": epoch, "score": score, "path": checkpoint_path})
    updated_entries.sort(key=lambda entry: entry["score"], reverse=True)

    stale_entries = updated_entries[top_k:]
    updated_entries = updated_entries[:top_k]

    keep_paths = {entry["path"] for entry in updated_entries}
    for stale_entry in stale_entries:
        stale_path = stale_entry["path"]
        if stale_path not in keep_paths and os.path.exists(stale_path):
            os.remove(stale_path)

    summary = _refresh_topk_aliases(best_model_dir, updated_entries, top_k=top_k)
    return updated_entries, summary

def load_model(args, model_best_or_final="best"):
    exp_save_dir= args.exp_save_dir
    model = build_model(args, input_channel=args.input_channel, num_classes=args.num_classes).to(device)
    if model_best_or_final == "best":
        candidate_paths = [
            os.path.join(exp_save_dir, 'best_models', 'checkpoint_top1.pth'),
            os.path.join(exp_save_dir, 'checkpoint_best.pth'),
        ]
    else:
        candidate_paths = [
            os.path.join(exp_save_dir, 'checkpoint_last.pth'),
            os.path.join(exp_save_dir, 'checkpoint_final.pth'),
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

    model.load_state_dict(state_dict)

    model.to(device)

    return model, model_path

def zero_shot(args,logger,model=None):
    valloader = getZeroShotDataloader(args)
    if model is None:
        model,model_path = load_model(args)

    logger.info("train file dir:{} val file dir:{}".format(args.train_file_dir, args.val_file_dir))
    criterion = losses.__dict__['BCEDiceLoss']().to(device)

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


def init_dir(args):
    exp_name = (args.exp_name or "default_exp").strip() or "default_exp"
    if args.output_dir:
        base_output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    else:
        base_output_dir = os.path.abspath('./output')

    if os.path.basename(os.path.normpath(base_output_dir)) == exp_name:
        exp_save_dir = base_output_dir
    else:
        exp_save_dir = os.path.join(base_output_dir, exp_name)

    os.makedirs(exp_save_dir, exist_ok=True)
    args.exp_save_dir = exp_save_dir

    log_dir = os.path.join(exp_save_dir, 'logs')
    config_dir = os.path.join(exp_save_dir, 'configs')
    best_model_dir = os.path.join(exp_save_dir, 'best_models')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)
    args.log_dir = log_dir
    args.config_dir = config_dir
    args.best_model_dir = best_model_dir

    config_file_path = os.path.join(config_dir, 'config.json')
    args_dict = vars(args)
    with open(config_file_path, 'w') as f:
        json.dump(args_dict, f, indent=4)
    print(f"Config saved to {config_file_path}")
    save_training_args(config_dir, args_dict)

    log_file = os.path.join(log_dir, 'training.log')
    logger = setup_logger(
        log_file=log_file,
        logger_name=f"uknee.main.{args.model}.{args.dataset_name}.{args.exp_name}",
    )
    history_writer = EpochLogWriter(log_dir)
    model = build_model(config=args,input_channel=args.input_channel, num_classes=args.num_classes).to(device)

    return exp_save_dir, log_dir, history_writer, logger, model


def validate(args,logger,model):
    trainloader,valloader = getDataloader(args)
    criterion = losses.__dict__['BCEDiceLoss']().to(device)
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
        for i_batch, sampled_batch in enumerate(valloader):
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

    val_metric_dict = {
        "val_loss":avg_meters['val_loss'].avg, "val_iou":avg_meters['val_iou'].avg, "val_SE":avg_meters['SE'].avg,
            "val_PC":avg_meters['PC'].avg, "val_F1":avg_meters['F1'].avg, "val_ACC":avg_meters['ACC'].avg
    }
    val_metric_dict = convert_to_numpy(val_metric_dict)
    return val_metric_dict



def train(args,exp_save_dir, log_dir, history_writer, logger, model):
    start_epoch = 0
    trainloader, valloader = getDataloader(args)
    best_model_dir = os.path.join(exp_save_dir, 'best_models')
    plot_path = os.path.join(exp_save_dir, 'training_dashboard.png')

    model_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"model:{args.model} model_parameters:{model_parameters}")
    logger.info(f"train file dir:{args.train_file_dir} val file dir:{args.val_file_dir}")
    logger.info(f"output dir:{exp_save_dir}")
    logger.info(f"{len(trainloader)} train iterations per epoch | {len(valloader)} validation iterations per epoch")
    
    optimizer, base_lr, optimizer_name = _build_optimizer(args, model, logger)
    criterion = losses.__dict__['BCEDiceLoss']().to(device)
    logger.info(f"optimizer:{optimizer_name} base_lr:{base_lr}")


    train_metric_dict = {
            "best_iou": 0,
            "best_epoch": 0,
            "best_iou_withSE": 0,
            "best_iou_withPC": 0,
            "best_iou_withF1": 0,
            "best_iou_withACC": 0,
            "last_iou": 0,
            "last_SE": 0,
            "last_PC": 0,
            "last_F1": 0,
            "last_ACC": 0
    }

    max_epoch = args.max_epochs
    max_iterations = max(len(trainloader) * max_epoch, 1)
    history_rows = []
    topk_entries = _load_topk_entries(best_model_dir)
    top_epochs = []
    top_model_summary = []
    last_lr = base_lr

    if args.resume:
        candidate_paths = [
            os.path.join(exp_save_dir, 'checkpoint_last.pth'),
            os.path.join(exp_save_dir, 'checkpoint_final.pth'),
            os.path.join(exp_save_dir, 'checkpoint.pth'),
        ]
        checkpoint_path = next((path for path in candidate_paths if os.path.exists(path)), None)
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['state_dict'])
            if checkpoint.get('optimizer') is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = int(checkpoint['epoch'])
            train_metric_dict["best_iou"] = float(checkpoint.get('best_iou', 0.0))
            logger.info(f"Resuming training from epoch {start_epoch} with best IoU {train_metric_dict['best_iou']}")

    iter_num = start_epoch * len(trainloader)

    for epoch_num in range(start_epoch, max_epoch):
        model.train()
        epoch_id = epoch_num + 1
        epoch_start_time = time.time()
        avg_meters = {'loss': AverageMeter(),
                      'iou': AverageMeter(),
                      'val_loss': AverageMeter(),
                      'val_iou': AverageMeter(),
                      'SE': AverageMeter(),
                      'PC': AverageMeter(),
                      'F1': AverageMeter(),
                      'ACC': AverageMeter()
                      }

        current_lr = optimizer.param_groups[0]['lr']
        logger.info("=" * 96)
        logger.info(
            "Epoch [%d/%d] started | optimizer=%s | lr=%.8f | train_batches=%d | val_batches=%d",
            epoch_id,
            max_epoch,
            optimizer_name,
            current_lr,
            len(trainloader),
            len(valloader),
        )

        train_progress = tqdm(
            enumerate(trainloader),
            total=len(trainloader),
            desc=f"Epoch [{epoch_id}/{max_epoch}] Train",
            dynamic_ncols=True,
            leave=True,
        )
        for i_batch, sampled_batch in train_progress:
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)

            if args.do_deeps:
                outputs = model(volume_batch)
                loss = deep_supervision_loss(outputs=outputs,label_batch=label_batch,loss_metric=criterion)
                outputs=outputs[-1]
            else:
                outputs = model(volume_batch)
                loss = criterion(outputs, label_batch)

            if not torch.isfinite(outputs).all():
                train_progress.close()
                _raise_non_finite_error(logger, epoch_id, i_batch, "non-finite outputs", volume_batch, label_batch, outputs)
            if not torch.isfinite(loss):
                train_progress.close()
                _raise_non_finite_error(logger, epoch_id, i_batch, loss.detach(), volume_batch, label_batch, outputs)

            iou, dice, _, _, _, _, _ = get_metrics(outputs, label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num += 1
            last_lr = lr_
            avg_meters['loss'].update(loss.item(), volume_batch.size(0))
            avg_meters['iou'].update(iou, volume_batch.size(0))
            train_progress.set_postfix(
                loss=f"{avg_meters['loss'].avg:.4f}",
                iou=f"{avg_meters['iou'].avg:.4f}",
                lr=f"{last_lr:.2e}",
            )
        train_progress.close()

        model.eval()
        with torch.no_grad():
            val_progress = tqdm(
                enumerate(valloader),
                total=len(valloader),
                desc=f"Epoch [{epoch_id}/{max_epoch}] Val",
                dynamic_ncols=True,
                leave=True,
            )
            for i_batch, sampled_batch in val_progress:
                input, target = sampled_batch['image'], sampled_batch['label']
                input = input.to(device)
                target = target.to(device)
                output = model(input)
                output = output[-1] if args.do_deeps else output
                loss = criterion(output, target)

                if not torch.isfinite(output).all():
                    val_progress.close()
                    _raise_non_finite_error(logger, epoch_id, i_batch, "non-finite val outputs", input, target, output)
                if not torch.isfinite(loss):
                    val_progress.close()
                    _raise_non_finite_error(logger, epoch_id, i_batch, loss.detach(), input, target, output)
                
                iou, _, SE, PC, F1, _, ACC = get_metrics(output, target)
                avg_meters['val_loss'].update(loss.item(), input.size(0))
                avg_meters['val_iou'].update(iou, input.size(0))
                avg_meters['SE'].update(SE, input.size(0))
                avg_meters['PC'].update(PC, input.size(0))
                avg_meters['F1'].update(F1, input.size(0))
                avg_meters['ACC'].update(ACC, input.size(0))
                val_progress.set_postfix(
                    val_loss=f"{avg_meters['val_loss'].avg:.4f}",
                    val_iou=f"{avg_meters['val_iou'].avg:.4f}",
                    val_f1=f"{avg_meters['F1'].avg:.4f}",
                )
            val_progress.close()

        epoch_row = {
            "epoch": epoch_id,
            "lr": last_lr,
            "train_loss": _as_float(avg_meters['loss'].avg),
            "train_iou": _as_float(avg_meters['iou'].avg),
            "val_loss": _as_float(avg_meters['val_loss'].avg),
            "val_iou": _as_float(avg_meters['val_iou'].avg),
            "val_SE": _as_float(avg_meters['SE'].avg),
            "val_PC": _as_float(avg_meters['PC'].avg),
            "val_F1": _as_float(avg_meters['F1'].avg),
            "val_ACC": _as_float(avg_meters['ACC'].avg),
        }
        history_rows.append(epoch_row)
        history_writer.append(epoch_row)

        epoch_seconds = time.time() - epoch_start_time

        if epoch_row["val_iou"] > train_metric_dict["best_iou"]:
            train_metric_dict["best_iou"] = epoch_row["val_iou"]
            train_metric_dict["best_epoch"] = epoch_id
            train_metric_dict["best_iou_withSE"] = epoch_row["val_SE"]
            train_metric_dict["best_iou_withPC"] = epoch_row["val_PC"]
            train_metric_dict["best_iou_withF1"] = epoch_row["val_F1"]
            train_metric_dict["best_iou_withACC"] = epoch_row["val_ACC"]

        if epoch_num == max_epoch - 1:
            train_metric_dict["last_iou"] = epoch_row["val_iou"]
            train_metric_dict["last_SE"] = epoch_row["val_SE"]
            train_metric_dict["last_PC"] = epoch_row["val_PC"]
            train_metric_dict["last_F1"] = epoch_row["val_F1"]
            train_metric_dict["last_ACC"] = epoch_row["val_ACC"]

        checkpoint_path = os.path.join(exp_save_dir, 'checkpoint_last.pth')
        _save_checkpoint(
            checkpoint_path,
            args=args,
            model=model,
            optimizer=optimizer,
            epoch=epoch_id,
            best_iou=train_metric_dict["best_iou"],
            metrics=epoch_row,
        )

        topk_entries, top_model_summary = _maybe_save_topk_checkpoint(
            best_model_dir=best_model_dir,
            topk_entries=topk_entries,
            score=epoch_row["val_iou"],
            epoch=epoch_id,
            args=args,
            model=model,
            optimizer=optimizer,
            metrics=epoch_row,
            top_k=3,
        )

        generated_plot_path, top_epochs = plot_training_dashboard(
            log_dir=exp_save_dir,
            history_rows=history_rows,
            loss_keys=[
                ("train_loss", "Training Loss"),
                ("val_loss", "Validation Loss"),
            ],
            metric_keys=[
                ("train_iou", "Train IoU"),
                ("val_iou", "Val IoU"),
                ("val_F1", "Val F1"),
                ("val_SE", "Val Recall"),
                ("val_PC", "Val Precision"),
                ("val_ACC", "Val Accuracy"),
            ],
            ranking_key="val_iou",
            maximize=True,
            top_k=3,
            filename=os.path.basename(plot_path),
            title=f"{args.model} | {args.dataset_name} | {args.exp_name}",
        )

        history_writer.write_summary({
            "best_metrics": convert_to_numpy(train_metric_dict),
            "top_epochs": top_epochs,
            "top_models": top_model_summary,
            "plot_path": str(generated_plot_path) if generated_plot_path else "",
        })

        logger.info(
            "Epoch [%d/%d] finished in %.1fs | train_loss=%.6f | train_iou=%.4f | "
            "val_loss=%.6f | val_iou=%.4f | val_SE=%.4f | val_PC=%.4f | val_F1=%.4f | val_ACC=%.4f",
            epoch_id,
            max_epoch,
            epoch_seconds,
            epoch_row["train_loss"],
            epoch_row["train_iou"],
            epoch_row["val_loss"],
            epoch_row["val_iou"],
            epoch_row["val_SE"],
            epoch_row["val_PC"],
            epoch_row["val_F1"],
            epoch_row["val_ACC"],
        )
        if top_model_summary:
            logger.info(
                "Current top models: %s",
                ", ".join(
                    f"Top{item['rank']} epoch {item['epoch']} = {item['value']:.4f}"
                    for item in top_model_summary
                ),
            )


    train_metric_dict=convert_to_numpy(train_metric_dict)
    train_metric_dict["plot_path"] = str(plot_path)
    train_metric_dict["log_dir"] = log_dir
    train_metric_dict["best_model_dir"] = best_model_dir
    train_metric_dict["config_dir"] = args.config_dir
    if top_epochs:
        train_metric_dict["top1_epoch"] = top_epochs[0]["epoch"]
        train_metric_dict["top1_val_iou"] = top_epochs[0]["value"]
    if len(top_epochs) > 1:
        train_metric_dict["top2_epoch"] = top_epochs[1]["epoch"]
        train_metric_dict["top2_val_iou"] = top_epochs[1]["value"]
    if len(top_epochs) > 2:
        train_metric_dict["top3_epoch"] = top_epochs[2]["epoch"]
        train_metric_dict["top3_val_iou"] = top_epochs[2]["value"]
    history_writer.write_summary({
        "best_metrics": train_metric_dict,
        "top_epochs": top_epochs,
        "top_models": top_model_summary if 'top_model_summary' in locals() else [],
        "plot_path": str(plot_path),
    })
    logger.info(f"Training completed. Best IoU: {train_metric_dict['best_iou']}, Best Epoch: {train_metric_dict['best_epoch']}, Best SE: {train_metric_dict['best_iou_withSE']}, Best PC: {train_metric_dict['best_iou_withPC']}, Best F1: {train_metric_dict['best_iou_withF1']}, Best ACC: {train_metric_dict['best_iou_withACC']}")
    logger.info(f"Last IoU: {train_metric_dict['last_iou']}, Last SE: {train_metric_dict['last_SE']}, Last PC: {train_metric_dict['last_PC']}, Last F1: {train_metric_dict['last_F1']}, Last ACC: {train_metric_dict['last_ACC']}")
    if top_epochs:
        logger.info(
            "Top epochs by val_iou: %s",
            ", ".join(
                f"Top{item['rank']} epoch {item['epoch']} = {item['value']:.4f}"
                for item in top_epochs
            ),
        )

    return train_metric_dict






if __name__ == "__main__":

    
    print(f"\n=== Testing model: {args.model} ===")
    _validate_runtime_config(args)

    exp_save_dir, log_dir, history_writer, logger, model = init_dir(args)
    row_data=vars(args)


    if args.just_for_test:
        if args.zero_shot_dataset_name != "":
            csv_file = f"./result/result_{args.dataset_name}_2_{args.zero_shot_dataset_name}_test.csv"
        else:
            csv_file = f"./result/result_{args.dataset_name}_test.csv"

        ensure_parent_dir(csv_file)
        file_exists = os.path.isfile(csv_file)
        model, model_path = load_model(args, model_best_or_final="best")
        print(f"Just for test, skipping training. loading model form best checkpoint. Model loaded from {model_path}")
        val_metric_dict = validate(args,logger, model)
        if args.zero_shot_dataset_name !="":
            zeroshot_result=zero_shot(args,logger, model)
        else:
            zeroshot_result=None
        if val_metric_dict:
            row_data.update(val_metric_dict)
        if zeroshot_result:
            row_data.update(zeroshot_result)

        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row_data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_data)
        exit()
    try:
        csv_file = f"./result/result_{args.dataset_name}_train.csv"
        ensure_parent_dir(csv_file)
        file_exists = os.path.isfile(csv_file)
        train_metric_dict = train(args,exp_save_dir, log_dir, history_writer, logger, model)
        if args.zero_shot_dataset_name !="":
            zeroshot_result=zero_shot(args,logger, model)
        else:
            zeroshot_result=None
        if train_metric_dict:
            row_data.update(train_metric_dict)
        if zeroshot_result:
            row_data.update(zeroshot_result)
        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row_data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_data)
        print(f"Model {args.model} training finished successfully")
    except Exception as e:
        row_data.update({"Error": str(e)})
        error_row = row_data.copy()
        error_log_file = "./ERROR.log"
        error_log_exists = os.path.isfile(error_log_file)
        with open(error_log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=error_row.keys())
            if not error_log_exists:
                writer.writeheader()
            writer.writerow(error_row)
        logger.exception("Training failed with an exception.")
        traceback.print_exc()
        print(f"Model {args.model} failed: {str(e)}")
        raise SystemExit(1)
    
