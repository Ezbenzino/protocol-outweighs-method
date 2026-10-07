"""Diagnose nodule size vs annotation area to assess task tractability per bin."""
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

stat = defaultdict(lambda: {"n": 0, "union_px": [], "maj_px": [], "maj_empty": 0})
for line in open("data/splits/val.csv"):
    cid = line.strip()
    if not cid or cid == "case_id":
        continue
    p = f"data/processed/patches/{cid}.npz"
    if not os.path.exists(p):
        continue
    meta = json.load(open(f"data/processed/npz/{cid}.json"))
    votes = np.load(p)["votes"]  # (N, 128, 128) uint8 0..4
    for j, nod in enumerate(meta["nodules"]):
        d = float(nod["diameter_mm"])
        b = "micro" if d < 5 else "small" if d < 10 else "medium" if d < 30 else "large"
        v = votes[j]
        union = int((v >= 1).sum())
        maj = int((v >= 2).sum())
        stat[b]["n"] += 1
        stat[b]["union_px"].append(union)
        stat[b]["maj_px"].append(maj)
        if maj == 0:
            stat[b]["maj_empty"] += 1

print(f"{'bin':<8} {'n':>5} {'union_px(mean/med)':>22} {'maj_px(mean/med)':>20} {'maj_empty':>10}")
for b in ["micro", "small", "medium", "large"]:
    s = stat[b]
    up, mp = np.array(s["union_px"]), np.array(s["maj_px"])
    print(f"{b:<8} {s['n']:>5} {up.mean():>9.1f}/{np.median(up):>8.0f} "
          f"{mp.mean():>9.1f}/{np.median(mp):>8.0f} {s['maj_empty']:>10}")
