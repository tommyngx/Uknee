import numpy as np


def _as_bool_mask(array):
    return np.asarray(array).squeeze().astype(bool)


def _confusion_counts(pred, target):
    pred = _as_bool_mask(pred)
    target = _as_bool_mask(target)
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")

    tp = np.logical_and(pred, target).sum(dtype=np.int64)
    fp = np.logical_and(pred, np.logical_not(target)).sum(dtype=np.int64)
    fn = np.logical_and(np.logical_not(pred), target).sum(dtype=np.int64)
    tn = pred.size - tp - fp - fn
    return tp, fp, fn, tn


def dice_coefficient(pred, target, zero_division=0.0):
    tp, fp, fn, _ = _confusion_counts(pred, target)
    denom = 2 * tp + fp + fn
    if denom == 0:
        return float(zero_division)
    return float((2.0 * tp) / denom)


def jaccard_index(pred, target, zero_division=0.0):
    tp, fp, fn, _ = _confusion_counts(pred, target)
    denom = tp + fp + fn
    if denom == 0:
        return float(zero_division)
    return float(tp / denom)


def precision_score(pred, target, zero_division=0.0):
    tp, fp, _, _ = _confusion_counts(pred, target)
    denom = tp + fp
    if denom == 0:
        return float(zero_division)
    return float(tp / denom)


def recall_score(pred, target, zero_division=0.0):
    tp, _, fn, _ = _confusion_counts(pred, target)
    denom = tp + fn
    if denom == 0:
        return float(zero_division)
    return float(tp / denom)


def specificity_score(pred, target, zero_division=0.0):
    _, fp, _, tn = _confusion_counts(pred, target)
    denom = tn + fp
    if denom == 0:
        return float(zero_division)
    return float(tn / denom)


def accuracy_score(pred, target, zero_division=0.0):
    tp, fp, fn, tn = _confusion_counts(pred, target)
    total = tp + fp + fn + tn
    if total == 0:
        return float(zero_division)
    return float((tp + tn) / total)


def _surface_distances(pred, target, voxelspacing=None, connectivity=1):
    from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure

    pred = _as_bool_mask(pred)
    target = _as_bool_mask(target)
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")
    if not pred.any():
        raise ValueError("Prediction mask is empty.")
    if not target.any():
        raise ValueError("Target mask is empty.")

    footprint = generate_binary_structure(pred.ndim, connectivity)
    pred_border = np.logical_xor(
        pred,
        binary_erosion(pred, structure=footprint, border_value=0),
    )
    target_border = np.logical_xor(
        target,
        binary_erosion(target, structure=footprint, border_value=0),
    )
    distance_map = distance_transform_edt(~target_border, sampling=voxelspacing)
    return distance_map[pred_border]


def hd95(pred, target, voxelspacing=None, connectivity=1):
    pred = _as_bool_mask(pred)
    target = _as_bool_mask(target)
    if not pred.any() and not target.any():
        return 0.0
    if not pred.any() or not target.any():
        return float("inf")

    distances = np.concatenate(
        [
            _surface_distances(pred, target, voxelspacing, connectivity),
            _surface_distances(target, pred, voxelspacing, connectivity),
        ]
    )
    return float(np.percentile(distances, 95))


def assd(pred, target, voxelspacing=None, connectivity=1):
    pred = _as_bool_mask(pred)
    target = _as_bool_mask(target)
    if not pred.any() and not target.any():
        return 0.0
    if not pred.any() or not target.any():
        return float("inf")

    pred_to_target = _surface_distances(pred, target, voxelspacing, connectivity)
    target_to_pred = _surface_distances(target, pred, voxelspacing, connectivity)
    return float((pred_to_target.mean() + target_to_pred.mean()) / 2.0)
