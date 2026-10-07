"""One-time extraction of a per-nodule patch bank for fast training I/O.

Full-case NPZ files (~118 MB each) are slow to sample from: every dataset
access decompresses a whole CT volume just to read one slice. This script
pre-crops every nodule sample to a fixed BANK_SIZE patch and stores one
small NPZ per case, cutting per-sample I/O by ~3 orders of magnitude.

Patches are stored at BANK_SIZE (default 128) so any config with
patch_size <= BANK_SIZE (default 96, micro 64) center-crops from the bank
with results identical to cropping the original slice.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.utils import roi_center_crop


def process_case(job):
    npz_path, json_path, out_path, bank_size = job
    case_id = os.path.splitext(os.path.basename(json_path))[0]
    if os.path.exists(out_path):
        return case_id, 0, "skip"
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        nodules = meta.get("nodules", [])
        if not nodules:
            return case_id, 0, "empty"
        npz = np.load(npz_path)
        slices_arr = npz["slices"]      # decompress once per case
        cons_arr = npz["consensus"]
        imgs, votes = [], []
        for nod in nodules:
            sl = int(nod["slice"])
            center = (float(nod["cy"]), float(nod["cx"]))
            imgs.append(roi_center_crop(slices_arr[sl], center, bank_size))
            votes.append(roi_center_crop(cons_arr[sl], center, bank_size))
        np.savez(out_path,
                 images=np.stack(imgs).astype(np.uint8),
                 votes=np.stack(votes).astype(np.uint8))
        return case_id, len(imgs), "ok"
    except Exception as e:  # noqa: BLE001 - report per-case failures
        return case_id, 0, f"error: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", default="data/processed/npz")
    ap.add_argument("--out_dir", default="data/processed/patches")
    ap.add_argument("--bank_size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    jobs = []
    for fn in sorted(os.listdir(args.npz_dir)):
        if not fn.endswith(".npz"):
            continue
        case_id = fn[:-4]
        jobs.append((
            os.path.join(args.npz_dir, fn),
            os.path.join(args.npz_dir, f"{case_id}.json"),
            os.path.join(args.out_dir, f"{case_id}.npz"),
            args.bank_size,
        ))

    ok = skip = empty = 0
    errors = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, (case_id, n, status) in enumerate(ex.map(process_case, jobs, chunksize=4)):
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            elif status == "empty":
                empty += 1
            else:
                errors.append((case_id, status))
            if (i + 1) % 100 == 0:
                print(f"[{i + 1}/{len(jobs)}] ok={ok} skip={skip} empty={empty} "
                      f"err={len(errors)}", flush=True)

    print(f"bank done: ok={ok} skip={skip} empty={empty} errors={len(errors)} "
          f"bank_size={args.bank_size} -> {args.out_dir}")
    for case_id, status in errors:
        print(f"  ERROR {case_id}: {status}")


if __name__ == "__main__":
    main()
