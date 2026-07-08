import torch
import numpy as np

from utils.binary_metrics import (
    accuracy_score,
    dice_coefficient,
    precision_score,
    recall_score,
    specificity_score,
)

def get_metrics(output, target):
    num_classes = int(output.shape[1]) if output.ndim >= 2 else 1
    if num_classes > 1:
        return get_multiclass_metrics(output, target, num_classes=num_classes)

    output = torch.sigmoid(output).cpu().detach().numpy()
    target = target.cpu().detach().numpy()

    output = (output > 0.5).astype(np.uint8)
    target = target.astype(np.uint8)

    dice = dice_coefficient(output, target, zero_division=0.0)

    intersection = np.sum(output * target)
    union = np.sum(output) + np.sum(target) - intersection
    iou = intersection / union if union > 0 else 0

    SE = recall_score(output, target, zero_division=0.0)

    PC = precision_score(output, target, zero_division=0.0)

    SP = specificity_score(output, target, zero_division=0.0)

    ACC = accuracy_score(output, target, zero_division=0.0)

    F1 = 2 * (PC * SE) / (PC + SE) if (PC + SE) > 0 else 0

    return iou, dice, SE, PC, F1, SP, ACC


def _prepare_multiclass_target(target, num_classes):
    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]
    elif target.ndim == 4 and target.shape[1] == num_classes:
        target = torch.argmax(target, dim=1)
    return target.long()


def _safe_divide(numerator, denominator, zero_division=0.0):
    if denominator == 0:
        return zero_division
    return numerator / denominator


def get_multiclass_metrics(output, target, num_classes=None, include_background=False):
    num_classes = int(num_classes or output.shape[1])
    prediction = torch.argmax(output, dim=1).cpu().detach().numpy().astype(np.int64)
    target = _prepare_multiclass_target(target, num_classes).cpu().detach().numpy().astype(np.int64)

    class_ids = range(num_classes) if include_background else range(1, num_classes)
    iou_scores = []
    dice_scores = []
    recall_scores = []
    precision_scores = []
    specificity_scores = []
    f1_scores = []

    for class_id in class_ids:
        pred_class = prediction == class_id
        target_class = target == class_id

        tp = float(np.logical_and(pred_class, target_class).sum())
        fp = float(np.logical_and(pred_class, np.logical_not(target_class)).sum())
        fn = float(np.logical_and(np.logical_not(pred_class), target_class).sum())
        tn = float(np.logical_and(np.logical_not(pred_class), np.logical_not(target_class)).sum())

        union = tp + fp + fn
        if union == 0:
            continue

        iou = _safe_divide(tp, union)
        dice = _safe_divide(2.0 * tp, 2.0 * tp + fp + fn)
        recall = _safe_divide(tp, tp + fn)
        precision = _safe_divide(tp, tp + fp)
        specificity = _safe_divide(tn, tn + fp, zero_division=0.0)
        f1 = _safe_divide(2.0 * precision * recall, precision + recall)

        iou_scores.append(iou)
        dice_scores.append(dice)
        recall_scores.append(recall)
        precision_scores.append(precision)
        specificity_scores.append(specificity)
        f1_scores.append(f1)

    accuracy = float((prediction == target).mean()) if target.size else 0.0
    if not iou_scores:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, accuracy

    return (
        float(np.mean(iou_scores)),
        float(np.mean(dice_scores)),
        float(np.mean(recall_scores)),
        float(np.mean(precision_scores)),
        float(np.mean(f1_scores)),
        float(np.mean(specificity_scores)),
        accuracy,
    )

def dice_coef(output, target):
    output = torch.sigmoid(output).cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    output = (output > 0.5).astype(np.uint8)
    target = target.astype(np.uint8)
    dice = dice_coefficient(output, target, zero_division=0.0)
    return dice

def get_accuracy(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == np.max(GT)  
    corr = np.sum(SR == GT)  
    acc = float(corr) / float(SR.size) 
    return acc
