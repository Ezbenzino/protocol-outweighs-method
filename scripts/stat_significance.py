"""Statistical significance test: consensus vs union target 5-fold CV.

Reads best validation Dice (majority) from 5 consensus folds and 5 union
folds, then performs:
  1. Paired t-test
  2. Wilcoxon signed-rank test
Reports p-values and effect size (Cohen's d for paired data).

Usage:
    python scripts/stat_significance.py
"""
import os
import re
import numpy as np
from scipy import stats


RUNS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "outputs", "runs")


def extract_best_dice(run_name):
    """Extract best validation dice_majority from train.log."""
    log_path = os.path.join(RUNS_DIR, run_name, "logs", "train.log")
    if not os.path.exists(log_path):
        return None
    best = None
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = re.search(r"saved best.*dice_majority=([0-9.]+)", line)
            if m:
                best = float(m.group(1))
    return best


def main():
    # Consensus 5-fold (already trained)
    consensus_folds = [f"main_fold{i}" for i in range(5)]
    # Union 5-fold (being trained)
    union_folds = [f"union_fold{i}" for i in range(5)]

    consensus_dice = []
    union_dice = []

    print("=" * 60)
    print("Consensus vs Union Target: 5-Fold CV Statistical Test")
    print("=" * 60)

    print("\nConsensus target folds:")
    for name in consensus_folds:
        d = extract_best_dice(name)
        if d is not None:
            consensus_dice.append(d)
            print(f"  {name}: {d:.4f}")
        else:
            print(f"  {name}: NOT FOUND")

    print("\nUnion target folds:")
    for name in union_folds:
        d = extract_best_dice(name)
        if d is not None:
            union_dice.append(d)
            print(f"  {name}: {d:.4f}")
        else:
            print(f"  {name}: NOT FOUND (training may not be complete)")

    if len(consensus_dice) < 5 or len(union_dice) < 5:
        print("\nWARNING: Not all 5 folds complete for both groups.")
        print("Results below are based on available folds only.")

    n = min(len(consensus_dice), len(union_dice))
    if n < 2:
        print("\nNeed at least 2 folds per group for statistical test.")
        return

    c = np.array(consensus_dice[:n])
    u = np.array(union_dice[:n])
    diff = c - u

    print(f"\n--- Results (n={n} paired folds) ---")
    print(f"Consensus: mean={c.mean():.4f}, std={c.std(ddof=1):.4f}")
    print(f"Union:     mean={u.mean():.4f}, std={u.std(ddof=1):.4f}")
    print(f"Difference: mean={diff.mean():.4f}, std={diff.std(ddof=1):.4f}")
    print(f"Effect size (Cohen's d, paired): {diff.mean() / diff.std(ddof=1):.2f}")

    # Paired t-test
    t_stat, t_p = stats.ttest_rel(c, u)
    print(f"\nPaired t-test: t={t_stat:.3f}, p={t_p:.6f}")

    # Wilcoxon signed-rank test
    try:
        w_stat, w_p = stats.wilcoxon(c, u)
        print(f"Wilcoxon signed-rank: W={w_stat:.1f}, p={w_p:.6f}")
    except ValueError as e:
        print(f"Wilcoxon test failed: {e}")

    # Sign test (simple)
    n_pos = (diff > 0).sum()
    n_neg = (diff < 0).sum()
    print(f"\nSign test: {n_pos} positive, {n_neg} negative out of {n}")

    print("\n" + "=" * 60)
    if t_p < 0.05:
        print("CONCLUSION: Significant difference (p < 0.05).")
        print(f"  Consensus target outperforms union by {diff.mean():.4f} Dice points.")
    else:
        print("CONCLUSION: No significant difference at p < 0.05.")
    print("=" * 60)


if __name__ == "__main__":
    main()
