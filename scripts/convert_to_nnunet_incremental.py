"""Incremental conversion: only convert missing/corrupt NIfTI files.
Skips cases where both image and label already exist and are valid.
"""
import os
import json
import numpy as np
import SimpleITK as sitk

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ_DIR = os.path.join(_ROOT, "data", "processed", "npz")
SPLIT_DIR = os.path.join(_ROOT, "data", "splits")
OUT_DIR = os.path.join(_ROOT, "data", "nnunet", "Dataset101_LIDC")


def load_case(case_id):
    npz_path = os.path.join(NPZ_DIR, f"{case_id}.npz")
    npz = np.load(npz_path, mmap_mode="r")
    slices = np.array(npz["slices"])
    consensus = np.array(npz["consensus"])
    return slices, consensus


def convert_ct_to_hu(slices_uint8):
    return slices_uint8.astype(np.float32) / 255.0 * 1400.0 - 1000.0


def save_nifti(array, filepath, spacing=(1.0, 1.0, 1.0)):
    img = sitk.GetImageFromArray(array)
    img.SetSpacing(spacing)
    sitk.WriteImage(img, filepath)


def is_valid_nifti(filepath):
    if not os.path.exists(filepath):
        return False
    try:
        img = sitk.ReadImage(filepath)
        return img.GetSize() != (0, 0, 0)
    except Exception:
        return False


def read_csv(path):
    ids = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line and line != "case_id":
                ids.append(line)
    return ids


def main():
    images_tr = os.path.join(OUT_DIR, "imagesTr")
    labels_tr = os.path.join(OUT_DIR, "labelsTr")
    images_ts = os.path.join(OUT_DIR, "imagesTs")
    os.makedirs(images_tr, exist_ok=True)
    os.makedirs(labels_tr, exist_ok=True)
    os.makedirs(images_ts, exist_ok=True)

    train_ids = read_csv(os.path.join(SPLIT_DIR, "train.csv"))
    val_ids = read_csv(os.path.join(SPLIT_DIR, "val.csv"))
    test_ids = read_csv(os.path.join(SPLIT_DIR, "test.csv"))
    trainval_ids = train_ids + val_ids

    print(f"Train+Val: {len(trainval_ids)}, Test: {len(test_ids)}")

    # Convert missing training cases
    converted_tr = 0
    for i, case_id in enumerate(trainval_ids):
        case_name = f"case_{i:04d}"
        img_path = os.path.join(images_tr, f"{case_name}_0000.nii.gz")
        lbl_path = os.path.join(labels_tr, f"{case_name}.nii.gz")
        if is_valid_nifti(img_path) and is_valid_nifti(lbl_path):
            continue
        try:
            slices, consensus = load_case(case_id)
            ct_hu = convert_ct_to_hu(slices)
            label = (consensus >= 2).astype(np.uint8)
            save_nifti(ct_hu.astype(np.int16), img_path)
            save_nifti(label, lbl_path)
            converted_tr += 1
            print(f"  Converted train {case_name} ({case_id}), shape={slices.shape}")
        except Exception as e:
            print(f"  ERROR train {case_name} ({case_id}): {e}")

    # Convert missing test cases
    converted_ts = 0
    for i, case_id in enumerate(test_ids):
        case_name = f"case_{i:04d}"
        img_path = os.path.join(images_ts, f"{case_name}_0000.nii.gz")
        if is_valid_nifti(img_path):
            continue
        try:
            slices, consensus = load_case(case_id)
            ct_hu = convert_ct_to_hu(slices)
            save_nifti(ct_hu.astype(np.int16), img_path)
            converted_ts += 1
            print(f"  Converted test {case_name} ({case_id}), shape={slices.shape}")
        except Exception as e:
            print(f"  ERROR test {case_name} ({case_id}): {e}")

    # Update dataset.json with correct format
    num_tr = len([f for f in os.listdir(images_tr) if f.endswith('.nii.gz')])
    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "nodule": 1},
        "numTraining": num_tr,
        "file_ending": ".nii.gz",
    }
    with open(os.path.join(OUT_DIR, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"\nIncremental conversion complete!")
    print(f"  New training: {converted_tr}, New test: {converted_ts}")
    print(f"  Total training images: {num_tr}")


if __name__ == "__main__":
    main()
