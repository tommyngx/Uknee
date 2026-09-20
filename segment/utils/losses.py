import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '5'
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    'one_hot', 'BCEDiceLoss', 'DiceLoss', 'DiceCELoss',
    'OsteophyteFocalTverskyCELoss',
]


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
            if target.ndim == inputs.ndim and target.size(1) == 1:
                target = target[:, 0]
            elif target.ndim == inputs.ndim and target.size(1) == inputs.shape[1]:
                target = torch.argmax(target, dim=1)
            ce_loss = self.ce(inputs, target.long())
            dice_loss = self.dice(inputs, target, softmax=True)
        return self.lambda_ce * ce_loss + self.lambda_dice * dice_loss


class OsteophyteFocalTverskyCELoss(nn.Module):
    """Cross-entropy for every class plus Focal Tversky on selected channels."""

    def __init__(
        self,
        n_classes,
        osteophyte_class_ids=(6, 7, 8, 9),
        fp_weight=0.30,
        fn_weight=0.70,
        gamma=1.30,
        smooth=1e-6,
        lambda_ft=1.0,
    ):
        super().__init__()
        self.n_classes = int(n_classes)
        self.osteophyte_class_ids = tuple(int(value) for value in osteophyte_class_ids)
        if not self.osteophyte_class_ids:
            raise ValueError("osteophyte_class_ids must not be empty")
        if min(self.osteophyte_class_ids) < 0 or max(self.osteophyte_class_ids) >= self.n_classes:
            raise ValueError(
                f"osteophyte_class_ids={self.osteophyte_class_ids} must be within "
                f"[0, {self.n_classes - 1}]"
            )
        self.fp_weight = float(fp_weight)
        self.fn_weight = float(fn_weight)
        self.gamma = float(gamma)
        self.smooth = float(smooth)
        self.lambda_ft = float(lambda_ft)
        if self.fp_weight < 0 or self.fn_weight < 0:
            raise ValueError("Focal Tversky FP/FN weights must be non-negative")
        if self.gamma <= 0 or self.smooth <= 0 or self.lambda_ft < 0:
            raise ValueError("gamma and smooth must be positive; lambda_ft must be non-negative")

    def focal_tversky(self, inputs, target):
        probabilities = torch.softmax(inputs, dim=1)
        targets = one_hot(target, self.n_classes).to(device=inputs.device, dtype=inputs.dtype)
        channels = torch.as_tensor(
            self.osteophyte_class_ids, device=inputs.device, dtype=torch.long
        )
        probabilities = probabilities.index_select(1, channels)
        targets = targets.index_select(1, channels)
        reduce_dims = tuple(range(2, inputs.ndim))
        true_positive = (probabilities * targets).sum(dim=reduce_dims)
        false_positive = (probabilities * (1.0 - targets)).sum(dim=reduce_dims)
        false_negative = ((1.0 - probabilities) * targets).sum(dim=reduce_dims)
        score = (true_positive + self.smooth) / (
            true_positive
            + self.fp_weight * false_positive
            + self.fn_weight * false_negative
            + self.smooth
        )
        positive_loss = (1.0 - score).pow(self.gamma)
        spatial_elements = 1
        for size in inputs.shape[2:]:
            spatial_elements *= int(size)
        # A target-empty channel has no FN term. Use its mean predicted
        # probability as an explicit, stable false-positive penalty rather
        # than ignoring the channel or relying on a smoothing-dominated ratio.
        negative_loss = (
            self.fp_weight * false_positive / max(spatial_elements, 1)
        ).pow(self.gamma)
        has_positive_target = targets.sum(dim=reduce_dims) > 0
        return torch.where(has_positive_target, positive_loss, negative_loss).mean()

    def forward(self, inputs, target):
        if inputs.shape[1] != self.n_classes:
            raise ValueError(
                f"Expected {self.n_classes} logit channels, received {inputs.shape[1]}"
            )
        if target.ndim == inputs.ndim and target.size(1) == 1:
            target = target[:, 0]
        elif target.ndim == inputs.ndim and target.size(1) == inputs.shape[1]:
            target = torch.argmax(target, dim=1)
        target = target.long()
        base_loss = F.cross_entropy(inputs, target)
        return base_loss + self.lambda_ft * self.focal_tversky(inputs, target)


def compute_kl_loss(p, q):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1),
                      F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1),
                      F.softmax(p, dim=-1), reduction='none')

    p_loss = p_loss.mean()
    q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2
    return loss
