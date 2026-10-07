"""Dataset sanity test using a tiny synthetic case (lean npz format)."""
import json
import os

import numpy as np

from src.data.dataset import LungNoduleDataset
from src.utils.config import AttrDict


def _make_case(tmp_path, case_id="case1"):
    H = W = 64
    N = 3
    slices = np.zeros((N, H, W), dtype=np.uint8)
    consensus = np.zeros((N, H, W), dtype=np.uint8)
    slices[1] = 100
    consensus[1, 20:30, 20:30] = 4   # all 4 raters agree -> vote=4

    np.savez(os.path.join(str(tmp_path), f"{case_id}.npz"),
             slices=slices, consensus=consensus)
    meta = {"nodules": [{"slice": 1, "cy": 25, "cx": 25, "diameter_mm": 4.0}]}
    with open(os.path.join(str(tmp_path), f"{case_id}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    with open(os.path.join(str(tmp_path), "train.csv"), "w", encoding="utf-8") as f:
        f.write("case_id\ncase1\n")


def test_sample_shape(tmp_path):
    _make_case(tmp_path)
    cfg = AttrDict({"data": {"patch_size": 32}, "model": {"in_channels": 3}})
    ds = LungNoduleDataset(str(tmp_path), os.path.join(str(tmp_path), "train.csv"),
                           cfg, phase="val")
    s = ds[0]
    assert s["image"].shape == (3, 32, 32)
    assert s["union_mask"].shape == (1, 32, 32)
    assert s["instance_ids"].shape == (32, 32)
    assert float(s["diameter_mm"].item()) == 4.0
    # vote=4 -> soft label 1.0 and disagreement 0 in the nodule
    assert abs(float(s["consensus"].max()) - 1.0) < 1e-6
    assert float(s["disagreement"].max()) < 1e-6
