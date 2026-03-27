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
