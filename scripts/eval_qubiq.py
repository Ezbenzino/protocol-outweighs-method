# QUBIQ 官方 Q-Dice 指标（论文附录用）
# 定义（QUBIQ challenge, arXiv:2405.18435, grand-challenge.org/participation）：
#   连续 GT = 多标注者平均 = consensus = V/N（"averaging multiple experts' annotations"）
#   预测概率图 p = region_prob(model, image)（sigmoid 输出，0~1）
#   Q-Dice = mean over thr in {0.1, 0.2, ..., 0.9} of Dice(gt>=thr, pred>=thr)
# 与 LIDC 主实验的"评测靶"（V>=k 硬掩膜）不同：Q-Dice 是阈值无关的连续指标，
# 直接用固定阈值网格平均，无需"标定最优阈值"，因此 val/test 划分均可直接评测。
#
# 用法（PowerShell，在仓库根目录下）：
#   # 5 折 val（每折模型测对应 val 折）
#   python scripts\eval_qubiq.py --config configs\qubiq\kidney_B.yaml --base configs\default.yaml ^
#       --folds 0,1,2,3,4 --prefixes qubiq_kidney_A,qubiq_kidney_B,qubiq_kidney_C --n-raters 3 ^
#       --out-dir outputs\analysis\qubiq_kidney_qdice
#   # 保留测试集（全部 5 折模型都测 test.csv）
#   python scripts\eval_qubiq.py --config configs\qubiq\kidney_B.yaml --base configs\default.yaml ^
#       --folds 0,1,2,3,4 --prefixes qubiq_kidney_A,qubiq_kidney_B,qubiq_kidney_C --n-raters 3 ^
#       --test --out-dir outputs\analysis\qubiq_kidney_qdice
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import LungNoduleDataset
from src.models.from_ckpt import load_for_eval, region_prob
from src.utils.config import load_config

# QUBIQ 官方阈值网格：0.1~0.9（9 个概率水平）
DEFAULT_THRS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


@torch.no_grad()
def eval_qdice_one(run_name, split, ckpt_path, loader, cfg, device, thrs):
    """对单个模型算每样本 Q-Dice（连续 GT=consensus=V/N，预测=概率图）。"""
    model, arch, act = load_for_eval(ckpt_path, cfg)
    model.to(device).eval()

    thrs_np = np.asarray(thrs, dtype=np.float32)
    rows = []
    t0, n_seen = time.time(), 0

    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        prob = region_prob(model, image).cpu().numpy()
        gt_soft = batch["consensus"].numpy()          # V/N，即多标注者平均（官方连续 GT）
        case_ids, dias = batch["case_id"], batch["diameter_mm"].numpy()

        for b in range(prob.shape[0]):
            p = prob[b, 0].astype(np.float32)
            g = gt_soft[b, 0].astype(np.float32)

            # 阈值网格向量化：每阈值算 Dice(gt>=t, pred>=t)
            g_bin = (g[None] > thrs_np[:, None, None]).astype(np.float32)    # (T,H,W)
            p_bin = (p[None] > thrs_np[:, None, None]).astype(np.float32)    # (T,H,W)
            inter = (g_bin * p_bin).reshape(len(thrs_np), -1).sum(1)         # (T,)
            denom = g_bin.reshape(len(thrs_np), -1).sum(1) + p_bin.reshape(len(thrs_np), -1).sum(1)
            dice_t = np.where(denom > 0, 2.0 * inter / np.maximum(denom, 1e-6), 1.0)
            qd = float(dice_t.mean())                                        # 阈值平均 = Q-Dice

            rows.append({
                "run": run_name, "fold": split, "case_id": case_ids[b],
                "diameter_mm": float(dias[b]), "qdice": round(qd, 4),
            })
            n_seen += 1

    dt = time.time() - t0
    print(f"  [{run_name}] arch={arch}  {n_seen} 样本 / {dt:.1f}s "
          f"({dt / max(n_seen, 1) * 1000:.1f} ms per sample)", flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def run_split(cfg, device, split, runs, ckpt_name, out_dir, batch_size, thrs):
    split_csv = os.path.join(cfg.data.split_dir, f"{split}.csv")
    if not os.path.exists(split_csv):
        print(f"[跳过] 找不到 {split_csv}")
        return None
    ds = LungNoduleDataset(cfg.data.npz_dir, split_csv, cfg, phase="val")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"\n=== split={split}  样本数={len(ds)}  Q-Dice 阈值={len(thrs)} 个 ===")

    all_rows = []
    for run in runs:
        ckpt = os.path.join(cfg.outputs.root, "runs", run, "checkpoints", ckpt_name)
        if not os.path.exists(ckpt):
            print(f"  [跳过] 找不到 {ckpt}")
            continue
        all_rows += eval_qdice_one(run, split, ckpt, loader, cfg, device, thrs)
    if not all_rows:
        return None
    os.makedirs(out_dir, exist_ok=True)
    fp = os.path.join(out_dir, f"qdice_{split}.csv")
    with open(fp, "w", encoding="utf-8", newline="") as fh:
        cols = ["run", "fold", "case_id", "diameter_mm", "qdice"]
        fh.write(",".join(cols) + "\n")
        for r in all_rows:
            fh.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"  -> {fp}")
    return all_rows


def summarize(all_split_rows, prefixes, tag):
    """按训练臂汇总 Q-Dice（样本级均值±std）。"""
    import numpy as np
    print(f"\n### Q-Dice 汇总（{tag}）—— 连续 GT=多标注者平均，阈值 0.1~0.9 平均")
    for pre in prefixes:
        vals = [r["qdice"] for r in all_split_rows if r["run"].startswith(pre)]
        if not vals:
            print(f"  {pre}: 无数据")
            continue
        mean, std = float(np.mean(vals)), float(np.std(vals))
        n = len(vals)
        print(f"  {pre.ljust(20)}: Q-Dice = {mean:.4f} ± {std:.4f}  (n={n})")
    return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--base", default=None)
    ap.add_argument("--folds", default=None, help="逗号分隔折号，如 0,1,2,3,4")
    ap.add_argument("--prefixes", default="qubiq_kidney_A,qubiq_kidney_B,qubiq_kidney_C")
    ap.add_argument("--n-raters", type=int, default=3)
    ap.add_argument("--ckpt-name", default="best.pth")
    ap.add_argument("--thrs", default=None, help="Q-Dice 阈值网格，逗号分隔；默认 0.1~0.9")
    ap.add_argument("--out-dir", default="outputs/analysis/qubiq_kidney_qdice")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--test", action="store_true",
                    help="在保留 test.csv 上评测（全部 5 折模型都测，样本池合并）")
    args = ap.parse_args()

    cfg = load_config(args.config, args.base)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    thrs = [float(x) for x in args.thrs.split(",")] if args.thrs else DEFAULT_THRS
    prefixes = [p.strip() for p in args.prefixes.split(",") if p.strip()]
    print(f"device={device}  Q-Dice 阈值={thrs}")

    folds = [f.strip() for f in args.folds.split(",") if f.strip()] if args.folds else []
    all_rows = []
    # 5 折 val（每折模型测对应 val 折）
    if folds:
        for k in folds:
            runs = [f"{p}_fold{k}" for p in prefixes]
            rows = run_split(cfg, device, f"val_fold{k}", runs, args.ckpt_name,
                             args.out_dir, args.batch_size, thrs)
            if rows:
                all_rows += rows
    # 保留测试集（全部模型测 test）
    if args.test:
        runs = [f"{p}_fold{k}" for p in prefixes for k in folds]
        rows = run_split(cfg, device, "test", runs, args.ckpt_name,
                         args.out_dir, args.batch_size, thrs)
        if rows:
            all_rows += rows

    if not all_rows:
        raise SystemExit("没有可用的评测结果，请检查 --folds / --prefixes / 模型路径")
    if folds:
        val_rows = [r for r in all_rows if r["fold"] != "test"]
        summarize(val_rows, prefixes, "5 折 val")
    if args.test:
        test_rows = [r for r in all_rows if r["fold"] == "test"]
        summarize(test_rows, prefixes, "保留测试集 test")


if __name__ == "__main__":
    main()
