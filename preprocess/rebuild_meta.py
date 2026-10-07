"""Rebuild nodule metadata from MAJORITY (>=2 vote) connected components, >=3mm.

The original process_lidc.py defined nodules as UNION (>=1 vote) connected
components. That pulls in (a) <3mm detection-only marks and (b) single-rater
isolated pixels whose majority mask is empty and therefore unsegmentable, which
pollutes the micro bin and destabilizes validation metrics.

This re-extracts nodules as MAJORITY-vote components (>=2/4 raters) with
area >= MIN_AREA_PX (~3 mm diameter), so every nodule has a non-empty,
segmentable target. Only the per-case JSON metadata is rebuilt; the vote maps
in the npz are unchanged. Patch bank must be rebuilt afterward.
"""
import argparse
import glob
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from skimage.measure import label, regionprops

SPACING_EST = 0.6   # mm/px typical in-plane LIDC spacing; only affects size bins
MIN_AREA_PX = 20    # ~3 mm diameter at 0.6 mm/px


def rebuild(npz_path, out_json):
    npz = np.load(npz_path)
    cons = npz["consensus"]            # uint8 vote 0..4
    nodules = []
    for sl in range(cons.shape[0]):
        maj = cons[sl] >= 2
        if not maj.any():
            continue
        lbl = label(maj, connectivity=2)
        for rp in regionprops(lbl):
            if rp.area < MIN_AREA_PX:
                continue
            area_mm2 = rp.area * SPACING_EST * SPACING_EST
            nodules.append({
                "slice": sl,
                "cy": float(rp.centroid[0]),
                "cx": float(rp.centroid[1]),
                "diameter_mm": float(2.0 * np.sqrt(area_mm2 / np.pi)),
            })
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"nodules": nodules}, f)
    return len(nodules)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", default="data/processed/npz")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    jobs = []
    for fn in sorted(os.listdir(args.npz_dir)):
        if fn.endswith(".npz"):
            cid = fn[:-4]
            jobs.append((os.path.join(args.npz_dir, fn),
                         os.path.join(args.npz_dir, f"{cid}.json")))

    total = 0
    empty = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, n in enumerate(ex.map(lambda j: rebuild(*j), jobs)):
            total += n
            if n == 0:
                empty += 1
            if (i + 1) % 100 == 0:
                print(f"[{i + 1}/{len(jobs)}] total_nodules={total} empty_cases={empty}",
                      flush=True)
    print(f"done: {len(jobs)} cases, {total} nodules (>=3mm majority), "
          f"{empty} cases with no nodule")


if __name__ == "__main__":
    main()
