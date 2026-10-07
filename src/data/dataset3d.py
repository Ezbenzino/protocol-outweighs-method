"""3D Dataset for LIDC-IDRI lung nodule segmentation.

Loads volumetric patches (D x H x W) centered on each nodule centroid.
Unlike the 2D patch bank, this loads directly from the npz volumes and
crops 3D patches on the fly.

Each case npz contains:
  slices: (N_slices, H, W) uint8, lung-windowed CT
  consensus: (N_slices, H, W) uint8, vote count 0..4
Each case json contains:
  nodules: [{slice, cy, cx, diameter_mm}, ...]
"""
import json
import os
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data import utils as data_utils


class _CaseCache(OrderedDict):
    """LRU cache for memory-mapped npz files (3D volumes)."""

    def __init__(self, max_cases=64):
        super().__init__()
        self.max_cases = max_cases

    def get(self, case_id, data_dir):
        if case_id in self:
            self.move_to_end(case_id)
            return self[case_id]
        npz = np.load(os.path.join(data_dir, f"{case_id}.npz"), mmap_mode="r")
        self[case_id] = npz
        if len(self) > self.max_cases:
            self.popitem(last=False)
        return npz


class LungNoduleDataset3D(Dataset):
    """3D patch dataset. Crops D x H x W volumes centered on nodule centroids."""

    def __init__(self, data_dir, split_csv, cfg, phase="train"):
        self.data_dir = data_dir
        self.phase = phase
        self.cfg = cfg
        self.patch_size = cfg.data.patch_size  # H=W
        self.depth = getattr(cfg.data, "depth", 16)  # D
        self.in_channels = cfg.model.in_channels
        self.case_ids = self._load_cases(split_csv)
        self.samples = self._build_index()
        # Subsample training set for 3D baseline (loading is slow)
        max_samples = getattr(cfg.data, "max_samples_3d", 0)
        if phase == "train" and max_samples > 0 and len(self.samples) > max_samples:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(self.samples), max_samples, replace=False)
            self.samples = [self.samples[i] for i in sorted(idx)]
        self.cache = _CaseCache(max_cases=64)
        # Preload all samples into memory for faster training (3D crop is slow)
        self._preloaded = None
        if getattr(cfg.data, "preload_3d", False):
            print(f"  [3D Dataset] Preloading {len(self.samples)} samples into memory...")
            self._preloaded = []
            for i in range(len(self.samples)):
                self._preloaded.append(self._load_sample(i))
            print(f"  [3D Dataset] Preload complete.")

    def _load_cases(self, split_csv):
        ids = []
        with open(split_csv, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and line != "case_id":
                    ids.append(line)
        return ids

    def _build_index(self):
        samples = []
        for case_id in self.case_ids:
            meta_path = os.path.join(self.data_dir, f"{case_id}.json")
            if not os.path.exists(meta_path):
                continue
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            for j, nod in enumerate(meta.get("nodules", [])):
                samples.append({
                    "case_id": case_id,
                    "nodule_idx": j,
                    "slice": int(nod["slice"]),
                    "cy": float(nod["cy"]),
                    "cx": float(nod["cx"]),
                    "diameter_mm": float(nod["diameter_mm"]),
                })
        return samples

    def __len__(self):
        return len(self.samples)

    def _crop_3d(self, volume, center_slice, center_y, center_x, depth, h, w):
        """Crop a depth x h x w patch from a (N, H, W) volume with padding."""
        N, H, W = volume.shape
        d_half = depth // 2
        h_half = h // 2
        w_half = w // 2

        # Slice range with padding
        z_start = center_slice - d_half
        z_end = center_slice + d_half + (depth % 2)
        y_start = center_y - h_half
        y_end = center_y + h_half + (h % 2)
        x_start = center_x - w_half
        x_end = center_x + w_half + (w % 2)

        # Pad if out of bounds
        z_pad_before = max(0, -z_start)
        z_pad_after = max(0, z_end - N)
        y_pad_before = max(0, -y_start)
        y_pad_after = max(0, y_end - H)
        x_pad_before = max(0, -x_start)
        x_pad_after = max(0, x_end - W)

        z_start = max(0, z_start)
        z_end = min(N, z_end)
        y_start = max(0, y_start)
        y_end = min(H, y_end)
        x_start = max(0, x_start)
        x_end = min(W, x_end)

        crop = volume[z_start:z_end, y_start:y_end, x_start:x_end]

        # Pad to target size
        if any([z_pad_before, z_pad_after, y_pad_before, y_pad_after,
                x_pad_before, x_pad_after]):
            crop = np.pad(crop, (
                (z_pad_before, z_pad_after),
                (y_pad_before, y_pad_after),
                (x_pad_before, x_pad_after),
            ), mode="constant", constant_values=0)

        return crop

    def _load_sample(self, idx):
        s = self.samples[idx]
        npz = self.cache.get(s["case_id"], self.data_dir)

        slices = npz["slices"]  # (N, H, W) uint8
        consensus = npz["consensus"]  # (N, H, W) uint8 0..4

        # Convert centroid to int indices
        cy = int(round(s["cy"]))
        cx = int(round(s["cx"]))
        cslice = s["slice"]

        # Crop 3D patches
        image = self._crop_3d(slices, cslice, cy, cx,
                               self.depth, self.patch_size, self.patch_size)
        vote = self._crop_3d(consensus, cslice, cy, cx,
                              self.depth, self.patch_size, self.patch_size)

        image = image.astype(np.float32) / 255.0  # [0, 1]
        vote = vote.astype(np.float32)  # 0..4

        consensus_prob = vote / 4.0
        union = (vote >= 1).astype(np.float32)
        majority_mask = (vote >= 2).astype(np.float32)

        # Add channel dimension: (1, D, H, W)
        image = image[np.newaxis, ...]
        if self.in_channels == 3:
            image = np.repeat(image, 3, axis=0)

        return {
            "image": torch.from_numpy(np.ascontiguousarray(image)),
            "union_mask": torch.from_numpy(np.ascontiguousarray(union[np.newaxis, ...])),
            "majority_mask": torch.from_numpy(np.ascontiguousarray(majority_mask[np.newaxis, ...])),
            "consensus": torch.from_numpy(np.ascontiguousarray(consensus_prob[np.newaxis, ...])),
            "disagreement": torch.from_numpy(np.ascontiguousarray(
                (4.0 * consensus_prob * (1.0 - consensus_prob))[np.newaxis, ...])),
            "diameter_mm": torch.tensor(s["diameter_mm"], dtype=torch.float32),
            "case_id": s["case_id"],
        }

    def __getitem__(self, idx):
        if self._preloaded is not None:
            return self._preloaded[idx]
        return self._load_sample(idx)
