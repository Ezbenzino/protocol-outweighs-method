"""Aggregate all experiment results into a summary table.

Reads best validation dice_majority from each training run's train.log,
plus test metrics from evaluate.py outputs. Generates:
  1. Consensus vs Union 5-fold CV table + statistical test
  2. Ablation study table (removal from full model)
  3. Baseline comparison table (ResNet-18, plain U-Net, 3D U-Net, nnU-Net)
  4. Weight sweep results
"""
import os
import re
import json
import numpy as np
from scipy import stats

RUNS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "outputs", "runs")


def extract_best(run_name):
    """Extract best validation dice_majority and epoch from train.log."""
    log_path = os.path.join(RUNS_DIR, run_name, "logs", "train.log")
    if not os.path.exists(log_path):
        return None, None
    best_dice = None
    best_epoch = None
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = re.search(r"saved best \(epoch (\d+), dice_majority=([0-9.]+)\)", line)
            if m:
                best_epoch = int(m.group(1))
                best_dice = float(m.group(2))
    return best_dice, best_epoch


def extract_final_val(run_name):
    """Extract final validation metrics (last val line)."""
    log_path = os.path.join(RUNS_DIR, run_name, "logs", "train.log")
    if not os.path.exists(log_path):
        return {}
    metrics = {}
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "val |" in line:
                for key in ["dice", "dice_majority", "iou", "hd95", "assd",
                            "dice_micro", "dice_small", "dice_medium"]:
                    m = re.search(rf"{key}=([0-9.nan]+)", line)
                    if m:
                        try:
                            metrics[key] = float(m.group(1))
                        except ValueError:
                            metrics[key] = float("nan")
    return metrics


def main():
    print("=" * 70)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 70)

    # --- 1. Main model ---
    print("\n--- Main Model (consensus target, full loss) ---")
    dice, epoch = extract_best("main")
    print(f"  Best val dice_majority: {dice:.4f} (epoch {epoch})")
    test_path = os.path.join(RUNS_DIR, "..", "logs", "test_metrics.json")
    if os.path.exists(test_path):
        with open(test_path) as f:
            test = json.load(f)
        print(f"  Test dice_majority: {test['dice_majority']:.4f}")
        print(f"  Test dice(union): {test['dice']:.4f}")
        print(f"  Test HD95: {test['hd95']:.2f}")
        print(f"  Test micro(3-5mm): {test['dice_micro']:.4f}")

    # --- 2. Consensus vs Union 5-fold CV ---
    print("\n--- Consensus vs Union Target: 5-Fold CV ---")
    consensus_folds = []
    union_folds = []
    for i in range(5):
        cd, ce = extract_best(f"main_fold{i}")
        ud, ue = extract_best(f"union_fold{i}")
        consensus_folds.append(cd)
        union_folds.append(ud)
        print(f"  Fold {i}: consensus={cd:.4f}, union={ud:.4f}, diff={cd-ud:.4f}")

    c = np.array([x for x in consensus_folds if x is not None])
    u = np.array([x for x in union_folds if x is not None])
    n = min(len(c), len(u))
    if n >= 2:
        diff = c[:n] - u[:n]
        print(f"\n  Consensus: {c.mean():.4f} ± {c.std(ddof=1):.4f}")
        print(f"  Union:     {u.mean():.4f} ± {u.std(ddof=1):.4f}")
        print(f"  Difference: {diff.mean():.4f} ± {diff.std(ddof=1):.4f}")
        t_stat, t_p = stats.ttest_rel(c[:n], u[:n])
        print(f"  Paired t-test: t={t_stat:.3f}, p={t_p:.6f}")
        try:
            w_stat, w_p = stats.wilcoxon(c[:n], u[:n])
            print(f"  Wilcoxon: p={w_p:.6f}")
        except Exception as e:
            print(f"  Wilcoxon: {e}")

    # --- 3. Ablation study ---
    print("\n--- Ablation Study (removal from full model) ---")
    ablations = [
        ("main (full)", "main"),
        ("- SITL", "abl_sitl"),
        ("- CSL", "abl_csl"),
        ("- BRBC", "abl_brbc"),
        ("- Lovasz", "abl_lovasz"),
        ("- all aux (Dice+BCE)", "abl_baseline"),
        ("- consensus (union target)", "abl_uniontgt"),
        ("ResNet-18 backbone", "base_unet"),
    ]
    for label, name in ablations:
        dice, epoch = extract_best(name)
        if dice is not None:
            main_dice = extract_best("main")[0] or 0
            diff = dice - main_dice
            print(f"  {label:30s}: {dice:.4f} (epoch {epoch:2d})  diff={diff:+.4f}")

    # --- 4. Weight sweep ---
    print("\n--- Auxiliary Loss Weight Sweep ---")
    weight_runs = [
        ("BRBC=0.1, Lov=0.1", "weight_brbc01_lov01"),
        ("BRBC=0.1, Lov=0.5", "weight_brbc01_lov05"),
        ("BRBC=1.0, Lov=0.1", "weight_brbc10_lov01"),
        ("BRBC=1.0, Lov=0.5", "weight_brbc10_lov05"),
    ]
    for label, name in weight_runs:
        dice, epoch = extract_best(name)
        if dice is not None:
            print(f"  {label:25s}: {dice:.4f} (epoch {epoch})")
        else:
            print(f"  {label:25s}: NOT YET TRAINED")

    # --- 5. Baselines ---
    print("\n--- Baseline Models ---")
    baselines = [
        ("Plain U-Net (no pretrain)", "plain_unet"),
        ("3D U-Net", "unet3d"),
    ]
    for label, name in baselines:
        dice, epoch = extract_best(name)
        if dice is not None:
            print(f"  {label:30s}: {dice:.4f} (epoch {epoch})")
        else:
            print(f"  {label:30s}: NOT YET TRAINED")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
