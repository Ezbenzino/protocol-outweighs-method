"""Convert LIDC-IDRI npz data to nnU-Net format (3D NIfTI).

nnU-Net expects:
  DatasetXXX_NAME/
    imagesTr/
      case_0000_0000.nii.gz   (modality 0 = CT)
    labelsTr/
      case_0000.nii.gz         (integer labels: 0=background, 1=nodule)
    imagesTs/
      case_XXXX_0000.nii.gz
    dataset.json

Our npz stores:
  slices: (N, H, W) uint8, lung-windowed CT (HU [-1000,400] -> [0,255])
  consensus: (N, H, W) uint8, vote count 0..4

We convert:
  CT: pixel/255 * 1400 - 1000 -> HU values (float32, saved as int16 NIfTI)
  Label: consensus >= 2 (majority vote) -> binary {0, 1}

For fair comparison with our 2D method, we use the same train/val/test splits:
  - Train + Val -> imagesTr/labelsTr (nnU-Net does its own CV)
  - Test -> imagesTs (held-out)

Usage:
    python scripts/convert_to_nnunet.py --out_dir data/nnunet/Dataset101_LIDC
"""
import argparse
import os
import json

import numpy as np
import SimpleITK as sitk


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ_DIR = os.path.join(_ROOT, "data", "processed", "npz")
SPLIT_DIR = os.path.join(_ROOT, "data", "splits")


def load_case(case_id):
    """Load slices and consensus from npz."""
    npz_path = os.path.join(NPZ_DIR, f"{case_id}.npz")
    npz = np.load(npz_path, mmap_mode="r")
    slices = np.array(npz["slices"])  # (N, H, W) uint8
    consensus = np.array(npz["consensus"])  # (N, H, W) uint8 0..4
    return slices, consensus


def convert_ct_to_hu(slices_uint8):
    """Convert lung-windowed uint8 [0,255] back to HU [-1000, 400]."""
    return slices_uint8.astype(np.float32) / 255.0 * 1400.0 - 1000.0


def save_nifti(array, filepath, spacing=(1.0, 1.0, 1.0)):
    """Save 3D numpy array as NIfTI. array shape (D, H, W) -> NIfTI (H, W, D)."""
    # SimpleITK expects (H, W, D) order for sitk.GetImageFromArray
    img = sitk.GetImageFromArray(array)
    img.SetSpacing(spacing)
    sitk.WriteImage(img, filepath)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, help="nnU-Net dataset directory")
    ap.add_argument("--dataset_id", type=int, default=101)
    ap.add_argument("--dataset_name", default="LIDC_Nodule")
    args = ap.parse_args()

    out_dir = args.out_dir
    images_tr = os.path.join(out_dir, "imagesTr")
    labels_tr = os.path.join(out_dir, "labelsTr")
    images_ts = os.path.join(out_dir, "imagesTs")
    os.makedirs(images_tr, exist_ok=True)
    os.makedirs(labels_tr, exist_ok=True)
    os.makedirs(images_ts, exist_ok=True)

    # Load splits
    def read_csv(path):
        ids = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line and line != "case_id":
                    ids.append(line)
        return ids

    train_ids = read_csv(os.path.join(SPLIT_DIR, "train.csv"))
    val_ids = read_csv(os.path.join(SPLIT_DIR, "val.csv"))
    test_ids = read_csv(os.path.join(SPLIT_DIR, "test.csv"))

    # Train + Val -> training set (nnU-Net does internal CV)
    trainval_ids = train_ids + val_ids
    print(f"Train+Val cases: {len(trainval_ids)}")
    print(f"Test cases: {len(test_ids)}")

    # Convert training cases
    for i, case_id in enumerate(trainval_ids):
        try:
            slices, consensus = load_case(case_id)
            ct_hu = convert_ct_to_hu(slices)
            label = (consensus >= 2).astype(np.uint8)  # majority vote

            # nnU-Net naming: case_0000
            case_name = f"case_{i:04d}"
            save_nifti(ct_hu.astype(np.int16),
                       os.path.join(images_tr, f"{case_name}_0000.nii.gz"))
            save_nifti(label, os.path.join(labels_tr, f"{case_name}.nii.gz"))

            if (i + 1) % 50 == 0:
                print(f"  Converted {i+1}/{len(trainval_ids)} training cases")
        except Exception as e:
            print(f"  ERROR {case_id}: {e}")

    # Convert test cases
    for i, case_id in enumerate(test_ids):
        try:
            slices, consensus = load_case(case_id)
            ct_hu = convert_ct_to_hu(slices)

            case_name = f"case_{i:04d}"
            save_nifti(ct_hu.astype(np.int16),
                       os.path.join(images_ts, f"{case_name}_0000.nii.gz"))
        except Exception as e:
            print(f"  ERROR test {case_id}: {e}")

    # Write dataset.json
    dataset_json = {
        "name": args.dataset_name,
        "description": "LIDC-IDRI pulmonary nodule segmentation (majority vote label)",
        "tensorImageSize": "3D",
        "reference": "LIDC-IDRI dataset",
        "licence": "CC BY 3.0",
        "release": "1.0",
        "modality": {"0": "CT"},
        "labels": {"0": "background", "1": "nodule"},
        "numTraining": len(trainval_ids),
        "numTest": len(test_ids),
        "file_ending": ".nii.gz",
    }
    with open(os.path.join(out_dir, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"\nConversion complete!")
    print(f"  Dataset: {out_dir}")
    print(f"  Training cases: {len(trainval_ids)}")
    print(f"  Test cases: {len(test_ids)}")
    print(f"  dataset.json written")


if __name__ == "__main__":
    main()
