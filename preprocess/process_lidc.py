"""Streaming preprocessing: DICOM -> uint8 windowed slices + consensus vote map -> npz.

One pass per case (no intermediate full-volume npy), keeping peak disk usage low.
The npz stores ONLY:
  slices    uint8 (N,H,W)  lung-windowed CT
  consensus uint8 (N,H,W)  vote count 0..4 from the (up to) 4 raters
Everything else (union mask, disagreement, instance ids, boundary) is derived
on the fly in the Dataset at load time.

DICOM series are matched to annotations via SeriesInstanceUID (read from DICOM
header 0020|000E and from each XML <SeriesInstanceUid>).

Also writes {case}.json with per-nodule {slice, cy, cx, diameter_mm}.

Usage:
  python preprocess/process_lidc.py --raw-dir data/raw/LIDC-IDRI \
      --annotations data/processed/annotations.json --out-dir data/processed/npz \
      [--delete-raw] [--workers 4] [--limit 5]
"""
import argparse
import concurrent.futures
import glob
import json
import os
import re
import shutil

import numpy as np
import SimpleITK as sitk
from skimage.draw import polygon
from skimage.measure import label, regionprops

HU_MIN, HU_MAX = -1000, 400


def extract_case_id(path):
    m = re.search(r"LIDC-IDRI-\d{4}", path)
    return m.group(0) if m else os.path.basename(path.rstrip("/\\"))


def read_series(case_dir):
    """Return (volume_hu float32, sop_uids, spacing, series_uid) sorted by z."""
    files = sorted(glob.glob(os.path.join(case_dir, "**", "*.dcm"), recursive=True))
    entries = []
    spacing = None
    series_uid = None
    for f in files:
        try:
            img = sitk.ReadImage(f)
        except Exception:
            continue
        if img.GetDimension() < 2:
            continue
        if series_uid is None and img.HasMetaDataKey("0020|000e"):
            series_uid = img.GetMetaData("0020|000e")
        z = float(img.GetOrigin()[2])
        sop = img.GetMetaData("0008|0018") if img.HasMetaDataKey("0008|0018") else ""
        arr = sitk.GetArrayFromImage(img)
        arr = arr[0] if arr.ndim == 3 else arr
        if spacing is None:
            spacing = [float(img.GetSpacing()[0]), float(img.GetSpacing()[1])]
        entries.append((z, sop, arr.astype(np.float32)))
    entries.sort(key=lambda e: e[0])
    if not entries:
        return None, None, None, None
    volume = np.stack([e[2] for e in entries], axis=0)
    sops = [e[1] for e in entries]
    return volume, sops, spacing, series_uid


def window_u8(volume):
    v = np.clip(volume, HU_MIN, HU_MAX)
    return ((v - HU_MIN) / (HU_MAX - HU_MIN) * 255.0).round().astype(np.uint8)


def rasterize(points, shape):
    rr, cc = polygon([p[1] for p in points], [p[0] for p in points], shape)
    m = np.zeros(shape, dtype=bool)
    m[rr, cc] = True
    return m


def build_consensus(ann, sops, shape):
    """Vote map (N,H,W uint8) from the reader sessions (one vote per reader)."""
    N, H, W = shape
    sop2idx = {sop: i for i, sop in enumerate(sops)}
    vote = np.zeros((N, H, W), dtype=np.uint8)
    for session in ann.get("sessions", []):
        doctor_vote = np.zeros((N, H, W), dtype=bool)
        for nod in session.get("nodules", []):
            for roi in nod.get("rois", []):
                idx = sop2idx.get(roi.get("sop_uid"))
                if idx is None:
                    continue
                doctor_vote[idx] |= rasterize(roi["points"], (H, W))
        vote += doctor_vote.astype(np.uint8)
    return np.clip(vote, 0, 4)


def nodule_meta(consensus, spacing):
    """Per-slice connected components of union (vote>=1) -> centroid + diameter."""
    nodules = []
    for i in range(consensus.shape[0]):
        union = consensus[i] >= 1
        if not union.any():
            continue
        lbl = label(union, connectivity=2)
        for rp in regionprops(lbl):
            area_mm2 = rp.area * spacing[0] * spacing[1]
            nodules.append({
                "slice": i,
                "cy": float(rp.centroid[0]),
                "cx": float(rp.centroid[1]),
                "diameter_mm": float(2.0 * np.sqrt(area_mm2 / np.pi)),
            })
    return nodules


def process_case(case_dir, anns, out_dir, delete_raw):
    case_id = extract_case_id(case_dir)
    npz_path = os.path.join(out_dir, f"{case_id}.npz")
    if os.path.exists(npz_path) and os.path.exists(os.path.join(out_dir, f"{case_id}.json")):
        return (case_id, "skip(exists)", 0)
    volume, sops, spacing, series_uid = read_series(case_dir)
    if volume is None:
        return (case_id, "skip(no-dicom)", 0)
    ann = anns.get(series_uid, {"sessions": []})
    slices = window_u8(volume)
    consensus = build_consensus(ann, sops, slices.shape)
    nodules = nodule_meta(consensus, spacing)
    np.savez(os.path.join(out_dir, f"{case_id}.npz"), slices=slices, consensus=consensus)
    with open(os.path.join(out_dir, f"{case_id}.json"), "w", encoding="utf-8") as f:
        json.dump({"nodules": nodules}, f)
    if delete_raw:
        shutil.rmtree(case_dir, ignore_errors=True)
    return (case_id, str(slices.shape), len(nodules))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--out-dir", default="data/processed/npz")
    ap.add_argument("--delete-raw", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="debug: only N cases")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.annotations, "r", encoding="utf-8") as f:
        anns = json.load(f)

    case_dirs = sorted(d for d in glob.glob(os.path.join(args.raw_dir, "*")) if os.path.isdir(d))
    # keep only case folders that actually contain DICOMs
    case_dirs = [d for d in case_dirs if glob.glob(os.path.join(d, "**", "*.dcm"), recursive=True)]
    if args.limit:
        case_dirs = case_dirs[: args.limit]

    print(f"processing {len(case_dirs)} cases with DICOM ...", flush=True)
    total = 0
    if args.workers <= 1:
        for d in case_dirs:
            cid, info, n = process_case(d, anns, args.out_dir, args.delete_raw)
            total += n
            print(f"[{cid}] {info} {n} nodules", flush=True)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_case, d, anns, args.out_dir, args.delete_raw): d for d in case_dirs}
            for fut in concurrent.futures.as_completed(futs):
                cid, info, n = fut.result()
                total += n
                print(f"[{cid}] {info} {n} nodules", flush=True)
    print(f"total nodule slice occurrences: {total}", flush=True)


if __name__ == "__main__":
    main()
