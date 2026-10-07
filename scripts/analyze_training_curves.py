"""P1-5: Training-curve analysis for the paper (4.8 / Discussion).

Parses every clean run's train.log and emits:
  - per-run best epoch / best dice_majority / early-stop epoch
  - per-arm epoch-by-epoch mean val dice_majority (fold-averaged)
  - consensus-vs-union (tgtC - tgtA) gap trajectory over epochs
  - train-loss trajectory (overfitting check)

Output: outputs/analysis/training_curves.json  +  console summary.
"""
import json
import os
import re
import sys

RUNS = os.path.join("outputs", "runs")
OUT_JSON = os.path.join("outputs", "analysis", "training_curves.json")

# arms that belong to the clean 4-target main table (ResNet-34 backbone)
MAIN_ARMS = ["tgtA", "tgtB", "tgtC", "tgtD"]


def parse_log(path):
    """Return dict: epochs=[...], train_loss=[...], dice_majority=[...],
    dice_05=[...] (raw dice), best_epoch, best_dice."""
    cur_epoch = None
    out = {"epochs": [], "train_loss": [], "dice_majority": [], "dice": []}
    re_ep = re.compile(r"Epoch (\d+)/(\d+) \| train_loss=([0-9.]+)")
    re_val = re.compile(r"\s+val \| dice=([0-9.]+), dice_majority=([0-9.]+)")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re_ep.search(line)
            if m:
                cur_epoch = int(m.group(1))
                out["epochs"].append(cur_epoch)
                out["train_loss"].append(float(m.group(3)))
                continue
            m = re_val.search(line)
            if m and cur_epoch is not None:
                out["dice"].append(float(m.group(1)))
                out["dice_majority"].append(float(m.group(2)))
                cur_epoch = None
    # best epoch (by dice_majority)
    if out["dice_majority"]:
        best_i = max(range(len(out["dice_majority"])), key=lambda i: out["dice_majority"][i])
        out["best_epoch"] = out["epochs"][best_i]
        out["best_dice"] = out["dice_majority"][best_i]
    return out


def main():
    runs = {}
    for name in sorted(os.listdir(RUNS)):
        log = os.path.join(RUNS, name, "logs", "train.log")
        if not os.path.isfile(log):
            continue
        parsed = parse_log(log)
        if not parsed["epochs"]:
            continue
        runs[name] = parsed

    # ---- fold-averaged per-arm curves over shared epoch grid ----
    arms = {}
    for name, p in runs.items():
        # only clean 4-target runs have explicit target in log; infer by prefix
        arm = name.split("_fold")[0].split("_s")[0]
        arms.setdefault(arm, []).append(p)

    grid = list(range(1, 61))
    arm_curve = {}
    for arm, ps in arms.items():
        if arm not in MAIN_ARMS:
            continue
        mean, n = {}, {}
        for p in ps:
            for ep, d in zip(p["epochs"], p["dice_majority"]):
                mean[ep] = mean.get(ep, 0.0) + d
                n[ep] = n.get(ep, 0) + 1
        arm_curve[arm] = [round(mean[e] / n[e], 4) if e in n else None for e in grid]

    # ---- gap trajectory tgtC - tgtA ----
    gap = []
    if "tgtC" in arm_curve and "tgtA" in arm_curve:
        for e, (c, a) in enumerate(zip(arm_curve["tgtC"], arm_curve["tgtA"])):
            if c is not None and a is not None:
                gap.append({"epoch": e + 1, "gap_pts": round((c - a) * 100, 2)})

    # ---- early-stop / best summary per run ----
    summary = []
    for name in sorted(runs):
        p = runs[name]
        ep_count = len(p["epochs"])
        # early stop inferred from last epoch recorded vs 60
        summary.append({
            "run": name,
            "epochs_run": ep_count,
            "early_stopped": ep_count < 60,
            "best_epoch": p.get("best_epoch"),
            "best_dice_majority": round(p.get("best_dice", float("nan")), 4),
        })

    result = {
        "grid_epochs": grid,
        "arm_curve_dice_majority": arm_curve,
        "consensus_minus_union_gap_pts": gap,
        "run_summary": summary,
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # ---- console report ----
    print("== 早停 / 最优轮次（主表 4 靶×5 折）==")
    hdr = f"{'run':<12}{'epochs':<8}{'best_ep':<8}{'best_dice':<10}"
    print(hdr)
    for s in summary:
        if s["run"].startswith(tuple(MAIN_ARMS)):
            print(f"{s['run']:<12}{s['epochs_run']:<8}{str(s['best_epoch']):<8}{s['best_dice_majority']:<10}")

    print("\n== 关键早期轮次 fold-平均 dice_majority（4 靶）==")
    sel = [1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 60]
    print(f"{'epoch':<7}" + "".join(f"{a:<9}" for a in MAIN_ARMS))
    for e in sel:
        row = f"{e:<7}"
        for a in MAIN_ARMS:
            v = arm_curve[a][e - 1]
            row += f"{('--' if v is None else round(v, 3)):<9}"
        print(row)

    print("\n== 共识-并集（tgtC-tgtA）差距轨迹（点）==")
    for g in gap:
        if g["epoch"] in [1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 60]:
            print(f"  epoch {g['epoch']:<3}: {g['gap_pts']:+.2f} pts")
    if gap:
        print(f"  最终稳定差距: {gap[-1]['gap_pts']:+.2f} pts (epoch {gap[-1]['epoch']})")
    print(f"\nJSON 已写入 {OUT_JSON}")


if __name__ == "__main__":
    main()
