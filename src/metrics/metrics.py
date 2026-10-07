"""Evaluation metrics: Dice, IoU, HD95, ASSD and size-stratified per-nodule Dice."""
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt
from skimage.measure import label as sklabel

from src.data.constants import SIZE_BINS_MM


def dice_score(pred, gt, smooth=1e-6):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = (pred & gt).sum()
    return 2.0 * inter / (pred.sum() + gt.sum() + smooth)


def instance_dice(pred, gt_instance, connectivity=2):
    """Dice between a GT instance and the predicted component overlapping it most.

    Penalizes both misses and false-positive area (unlike raw pixel coverage).
    """
    pred = pred.astype(bool)
    gt_instance = gt_instance.astype(bool)
    overlap = pred & gt_instance
    if overlap.sum() == 0:
        return 0.0
    lab, _ = sklabel(pred, connectivity=connectivity, return_num=True)
    best = 0.0
    for c in np.unique(lab[overlap]):
        if c == 0:
            continue
        comp = lab == c
        inter = (comp & gt_instance).sum()
        best = max(best, 2.0 * inter / (comp.sum() + gt_instance.sum() + 1e-6))
    return float(best)


def iou_score(pred, gt, smooth=1e-6):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return inter / (union + smooth)


def _surface(mask):
    return mask & ~binary_erosion(mask)


def surface_distances(pred, gt):
    """Return (hd, hd95, assd); NaN if either mask is empty."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if pred.sum() == 0 or gt.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    ps = _surface(pred)
    gs = _surface(gt)
    d2ps = distance_transform_edt(~ps)
    d2gs = distance_transform_edt(~gs)
    dists_gt = d2ps[gs]
    dists_pred = d2gs[ps]
    hd = max(dists_gt.max(), dists_pred.max())
    hd95 = max(np.percentile(dists_gt, 95), np.percentile(dists_pred, 95))
    assd = (dists_gt.sum() + dists_pred.sum()) / (len(dists_gt) + len(dists_pred))
    return float(hd), float(hd95), float(assd)


def size_bucket(diameter_mm, bins=SIZE_BINS_MM):
    if diameter_mm < bins[0]:
        return "micro"
    if diameter_mm < bins[1]:
        return "small"
    if diameter_mm < bins[2]:
        return "medium"
    return "large"
