"""QUBIQ -> npz + json preprocessing (external-dataset validation, P2-1).

Converts a QUBIQ sub-dataset (per-rater NIfTI labels) into the SAME format the
project uses for LIDC, so the whole controlled framework (train.py, eval_symmetric,
make_splits.py) works unchanged:

  {case}.npz  -> slices (uint8 windowed) + consensus (uint8 vote 0..N_raters)
  {case}.json -> per-region metadata [{slice, cy, cx, diameter_mm}]

QUBIQ data layout (grand-challenge.org; download requires joining the QUBIQ 2021
challenge and accepting its data-use terms — see manuscript Data Availability):
  <subdataset>/cases/<case>/   image.nii + <task>_<rater>.nii  (one label per rater)
  e.g. kidney: image.nii, task01_1.nii, task01_2.nii, task01_3.nii  (3 raters)
  e.g. brain-growth: image.nii, task01_1.nii .. task01_7.nii      (7 raters)

For 2D slices the nii is (H, W); for volumetric data it is (N, H, W) and every
slice is kept (background slices are fine — the Dataset's background sampler and
the split are designed around them).

Usage:
  python preprocess/build_qubiq.py --raw-dir data/raw/qubiq/kidney \
      --out-dir data/qubiq/kidney/npz --task task01 --n-raters 3 \
      --window humin=-150 humax=350 --workers 4

  MRI subdatasets (brain-growth/tumor, prostate): use --normalize minmax
  (per-volume min-max -> uint8) instead of an HU window.
"""
import argparse
import concurrent.futures
import glob
import json
import os
import re

import numpy as np
import SimpleITK as sitk
from skimage.measure import label, regionprops

IMAGE_PATTERNS = (r"image\.nii", r"img\.nii", r"case\d*\.nii")


def discover_files(case_dir, task):
    """Return (image_path, label_paths) sorted by rater number."""
    files = sorted(glob.glob(os.path.join(case_dir, "*.nii*")))
    if not files:
        files = sorted(glob.glob(os.path.join(case_dir, "**", "*.nii*"), recursive=True))
    image, labels = None, []
    for f in files:
        base = os.path.basename(f).lower()
        if any(re.search(p, base) for p in IMAGE_PATTERNS):
            image = f
            continue
        if task and task.lower() in base:
            labels.append(f)
        else:
            labels.append(f)  # fallback: everything not matched as image is a rater label
    if image is None and labels:
        # pick the file with the largest volume as the image
        image = max(labels, key=lambda f: np.prod(sitk.ReadImage(f).GetSize()))
        labels = [f for f in labels if f != image]
    labels.sort(key=lambda f: re.findall(r"\d+", os.path.basename(f)))
    return image, labels


def read_volume(path):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)          # (N,H,W) or (H,W)
    return np.asarray(arr, dtype=np.float32)


def process_case(case_dir, out_dir, task, n_raters, humin, humax, normalize):
    case_id = os.path.basename(case_dir.rstrip("/\\"))
    npz_path = os.path.join(out_dir, f"{case_id}.npz")
    json_path = os.path.join(out_dir, f"{case_id}.json")
    if os.path.exists(npz_path) and os.path.exists(json_path):
        return case_id, "skip(exists)", 0
    image_path, label_paths = discover_files(case_dir, task)
    if image_path is None or not label_paths:
        return case_id, f"skip(no files: img={image_path} labels={len(label_paths)})", 0
    image = read_volume(image_path)
    if image.ndim == 2:
        image = image[None, ...]
    N, H, W = image.shape
    # window / normalize -> uint8
    if normalize == "minmax":
        lo, hi = float(image.min()), float(image.max())
        denom = (hi - lo) if hi > lo else 1.0
        slices = (((image - lo) / denom) * 255.0).clip(0, 255).round().astype(np.uint8)
    else:
        v = np.clip(image, humin, humax)
        slices = ((v - humin) / (humax - humin) * 255.0).round().astype(np.uint8)
    # vote map from per-rater binary labels
    vote = np.zeros((N, H, W), dtype=np.uint8)
    used_raters = 0
    for lp in label_paths[:n_raters]:
        lab = read_volume(lp)
        if lab.ndim == 2:
            lab = lab[None, ...]
        if lab.shape != (N, H, W):
            # resample label to image grid if needed
            sitk_img = sitk.GetImageFromArray(lab)
            sitk_img.SetSpacing((1.0, 1.0, 1.0))
            ref = sitk.GetImageFromArray(np.zeros((N, H, W), dtype=np.uint8))
            ref.SetSpacing((1.0, 1.0, 1.0))
            lab = sitk.GetArrayFromImage(sitk.Resample(sitk_img, ref, sitk.Transform(),
                                                       sitk.sitkNearestNeighbor, 0.0))
            lab = np.asarray(lab)
        vote += (lab > 0).astype(np.uint8)
        used_raters += 1
    vote = np.clip(vote, 0, n_raters)
    # per-slice region metadata (union components)
    spacing = [1.0, 1.0]
    regions = []
    for i in range(N):
        union = vote[i] >= 1
        if not union.any():
            continue
        lbl = label(union, connectivity=2)
        for rp in regionprops(lbl):
            area_mm2 = rp.area * spacing[0] * spacing[1]
            regions.append({
                "slice": i,
                "cy": float(rp.centroid[0]),
                "cx": float(rp.centroid[1]),
                "diameter_mm": float(2.0 * np.sqrt(area_mm2 / np.pi)),
            })
    np.savez(npz_path, slices=slices, consensus=vote)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"n_raters": used_raters, "nodules": regions}, f, ensure_ascii=False)
    return case_id, f"{slices.shape} raters={used_raters}", len(regions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True, help="path to the QUBIQ subdataset (contains cases/)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--task", default="task01")
    ap.add_argument("--n-raters", type=int, default=3)
    ap.add_argument("--window", default="humin=-150 humax=350", nargs="+",
                    help="CT window: 'humin=-150 humax=350'. Ignored if --normalize minmax")
    ap.add_argument("--normalize", choices=["hu", "minmax"], default="hu",
                    help="hu: CT window; minmax: per-volume min-max (MRI)")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    humin, humax = -150.0, 350.0
    for kv in args.window:
        k, _, v = kv.partition("=")
        if k == "humin":
            humin = float(v)
        elif k == "humax":
            humax = float(v)

    cases_root = os.path.join(args.raw_dir, "cases")
    if not os.path.isdir(cases_root):
        cases_root = args.raw_dir
    case_dirs = sorted(d for d in glob.glob(os.path.join(cases_root, "*")) if os.path.isdir(d))
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"processing {len(case_dirs)} cases -> {args.out_dir} (task={args.task}, "
          f"raters={args.n_raters}, norm={args.normalize})", flush=True)

    jobs = [(d, args.out_dir, args.task, args.n_raters, humin, humax, args.normalize) for d in case_dirs]
    total = 0
    if args.workers <= 1:
        for d in case_dirs:
            cid, info, n = process_case(d, args.out_dir, args.task, args.n_raters,
                                        humin, humax, args.normalize)
            total += n
            print(f"[{cid}] {info} {n} regions", flush=True)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(process_case, *j) for j in jobs]
            for fut in concurrent.futures.as_completed(futs):
                cid, info, n = fut.result()
                total += n
                print(f"[{cid}] {info} {n} regions", flush=True)
    print(f"total region occurrences: {total}", flush=True)


if __name__ == "__main__":
    main()
