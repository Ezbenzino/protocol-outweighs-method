"""Data helper utilities: connected components, ROI crop, boundary extraction."""
import numpy as np
from scipy.ndimage import binary_erosion
from skimage.measure import label as sklabel


def label_instances(mask, connectivity=2):
    """Label connected components of a binary mask (0 = background)."""
    mask = np.asarray(mask) > 0
    if mask.ndim == 3:
        return np.stack([label_instances(m, connectivity) for m in mask], axis=0)
    lbl, _ = sklabel(mask, connectivity=connectivity, return_num=True)
    return lbl.astype(np.int64)


def boundary_from_mask(mask, width=1):
    """Extract a `width`-pixel boundary band from a binary mask."""
    mask = np.asarray(mask) > 0
    eroded = binary_erosion(mask, iterations=width)
    return (mask & ~eroded).astype(np.uint8)


def roi_center_crop(arr, center, size):
    """Crop a 2D/3D array to (size, size) around (cy, cx) with zero padding."""
    cy, cx = int(round(center[0])), int(round(center[1]))
    arr = np.asarray(arr)
    if arr.ndim == 2:
        H, W = arr.shape
        y0, x0 = cy - size // 2, cx - size // 2
        y1, x1 = y0 + size, x0 + size
        pad_top = max(0, -y0)
        pad_bottom = max(0, y1 - H)
        pad_left = max(0, -x0)
        pad_right = max(0, x1 - W)
        crop = arr[max(0, y0):min(H, y1), max(0, x0):min(W, x1)]
        if pad_top or pad_bottom or pad_left or pad_right:
            crop = np.pad(crop, ((pad_top, pad_bottom), (pad_left, pad_right)))
        return crop
    # (C, H, W)
    return np.stack([roi_center_crop(arr[c], center, size) for c in range(arr.shape[0])], axis=0)
