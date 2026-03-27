import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '5'
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['one_hot', 'BCEDiceLoss', 'DiceLoss', 'DiceCELoss']


def one_hot(target, num_classes):
    if target.ndim > 1 and target.size(1) == 1:
        target = target.squeeze(1)
    target = target.long()
    output = F.one_hot(target, num_classes=num_classes)
    dims = (0, output.ndim - 1, *range(1, output.ndim - 1))
    return output.permute(dims).float()


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


class DiceLoss(nn.Module):
    def __init__(self, n_classes, smooth=1e-5):
        super().__init__()
        self.n_classes = n_classes
        self.smooth = smooth

    def _binary_dice_loss(self, score, target):
        target = target.float()
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        return 1 - (2 * intersect + self.smooth) / (z_sum + y_sum + self.smooth)

    def _prepare_binary_target(self, target):
        if target.ndim == 3:
            target = target.unsqueeze(1)
        return target.float()

    def forward(self, inputs, target, weight=None, softmax=False, sigmoid=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        elif sigmoid or inputs.shape[1] == 1:
            inputs = torch.sigmoid(inputs)

        if inputs.shape[1] == 1:
            target = self._prepare_binary_target(target)
            return self._binary_dice_loss(inputs, target)

        target = one_hot(target, self.n_classes).to(inputs.device)
        if weight is None:
            weight = [1.0] * self.n_classes

        assert inputs.size() == target.size(), (
            f'predict {inputs.size()} & target {target.size()} shape do not match'
        )

        loss = 0.0
        for i in range(self.n_classes):
            loss += self._binary_dice_loss(inputs[:, i], target[:, i]) * weight[i]
        return loss / self.n_classes


class DiceCELoss(nn.Module):
    def __init__(self, n_classes=1, lambda_ce=0.3, lambda_dice=0.7):
        super().__init__()
        self.n_classes = n_classes
        self.lambda_ce = lambda_ce
        self.lambda_dice = lambda_dice
        self.dice = DiceLoss(n_classes=n_classes)
        self.ce = nn.CrossEntropyLoss() if n_classes > 1 else None

    def forward(self, inputs, target):
        if inputs.shape[1] == 1:
            if target.ndim == 3:
                target = target.unsqueeze(1)
            ce_loss = F.binary_cross_entropy_with_logits(inputs, target.float())
            dice_loss = self.dice(inputs, target, sigmoid=True)
        else:
            ce_loss = self.ce(inputs, target.long())
            dice_loss = self.dice(inputs, target, softmax=True)
        return self.lambda_ce * ce_loss + self.lambda_dice * dice_loss


def compute_kl_loss(p, q):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1),
                      F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1),
                      F.softmax(p, dim=-1), reduction='none')

    p_loss = p_loss.mean()
    q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2
    return loss
