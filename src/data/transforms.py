"""Data augmentation (numpy/scipy based, no external augmentation dependency)."""
import numpy as np
from scipy.ndimage import rotate, zoom

IMAGE_KEY = "image"
MASK_KEYS = ("union_mask", "majority_mask", "consensus", "disagreement",
             "instance_ids", "boundary", "maj_instance_ids", "maj_boundary")
ARRAY_KEYS = (IMAGE_KEY,) + MASK_KEYS


def _fit(a, target_shape):
    """Center crop or zero-pad a 2D array `a` back to `target_shape`."""
    H, W = a.shape
    tH, tW = target_shape
    if H > tH:
        y0 = (H - tH) // 2
        a = a[y0:y0 + tH]
    if W > tW:
        x0 = (W - tW) // 2
        a = a[:, x0:x0 + tW]
    if H < tH:
        py = tH - H
        top = py // 2
        a = np.pad(a, ((top, py - top), (0, 0)))
    if W < tW:
        px = tW - W
        left = px // 2
        a = np.pad(a, ((0, 0), (left, px - left)))
    return a


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if np.random.rand() < self.p:
            for k in ARRAY_KEYS:
                if k in sample:
                    sample[k] = np.ascontiguousarray(sample[k][..., ::-1])
        return sample


class RandomRotation:
    def __init__(self, degrees=10, p=0.5):
        self.degrees = degrees
        self.p = p

    def __call__(self, sample):
        if np.random.rand() < self.p:
            angle = np.random.uniform(-self.degrees, self.degrees)
            for k in ARRAY_KEYS:
                if k in sample:
                    order = 1 if k == IMAGE_KEY else 0
                    sample[k] = rotate(sample[k], angle, order=order, reshape=False, mode="nearest")
        return sample


class RandomScale:
    def __init__(self, scale_range=(0.9, 1.1), p=0.5):
        self.scale_range = scale_range
        self.p = p

    def __call__(self, sample):
        if np.random.rand() < self.p:
            s = np.random.uniform(*self.scale_range)
            target = sample[IMAGE_KEY].shape
            for k in ARRAY_KEYS:
                if k in sample:
                    order = 1 if k == IMAGE_KEY else 0
                    sample[k] = _fit(zoom(sample[k], s, order=order), target)
        return sample


class RandomBrightnessContrast:
    def __init__(self, brightness=0.2, contrast=0.2, p=0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.p = p

    def __call__(self, sample):
        if np.random.rand() < self.p and IMAGE_KEY in sample:
            img = sample[IMAGE_KEY]
            b = np.random.uniform(-self.brightness, self.brightness)
            c = np.random.uniform(1 - self.contrast, 1 + self.contrast)
            sample[IMAGE_KEY] = np.clip(img * c + b, 0.0, 1.0).astype(img.dtype)
        return sample


def build_transforms(cfg, phase):
    """Return a Compose for training, or None for eval (no augmentation)."""
    if phase == "train":
        return Compose([
            RandomHorizontalFlip(0.5),
            RandomRotation(10, 0.5),
            RandomScale((0.9, 1.1), 0.5),
            RandomBrightnessContrast(0.2, 0.2, 0.5),
        ])
    return None
