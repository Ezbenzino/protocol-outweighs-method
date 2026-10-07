"""对称评测 v2 —— 零训练成本诊断实验（评测与落盘，不做统计）。

v2 相对 v1 的改动（都是 v1 暴露出来的问题）：
1. 阈值范围扩到 0.05~0.95。v1 用 0.30~0.70，结果 union 组的 dice_v2 在
   上界 0.70 处仍在上升、main 组的 dice_v1 在下界 0.30 处仍在上升，
   两组的"最优阈值"都没被扫到，导致"各自最优下的公平对比"无法成立。
2. 逐样本落盘 GT 面积 gt_area_v1..v4。v1 里 dice_v3/dice_v4 把
   "GT 为空但预测非空"的样本记 0 分，混进均值里，数值不可信。
   有了 GT 面积就能在分析阶段只统计 GT 非空的样本。
3. 阈值循环向量化（einsum 一次算完 T×4 个 Dice），19 个阈值的耗时
   和 v1 的 9 个阈值基本持平。
4. --folds 模式：一条命令跑完 5 折 × N 组，自动配对 val_foldK。

用法（PowerShell，在仓库根目录下）：
    # 单折快速验证
    python scripts\\eval_symmetric.py --config configs\\default.yaml --folds 0
    # 全部 5 折
    python scripts\\eval_symmetric.py --config configs\\default.yaml --folds 0,1,2,3,4
"""
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

# GT 口径：V>=1 并集 / V>=2 多数 / ... / V>=N 全体一致（N = 标注者数，默认 4）
# LIDC=4；QUBIQ 肾脏=3、脑生长=7 等由 --n-raters 覆盖。


@torch.no_grad()
def eval_one(run_name, fold_tag, ckpt_path, loader, cfg, device, thr, n_raters=4):
    """一次推理算完所有阈值 × 所有 GT 口径。

    向量化说明：对每个样本，把 T 个阈值的二值预测堆成 (T,H,W)，
    N 个 GT 口径堆成 (N,H,W)，用一次 einsum 得到 (T,N) 的交集矩阵，
    再广播成 Dice。比 v1 的双重 for 循环快一个量级。
    """
    vote_levels = tuple(range(1, n_raters + 1))
    model, arch, act = load_for_eval(ckpt_path, cfg)
    model.to(device).eval()

    thr_np = np.asarray(thr, dtype=np.float32)
    rows, free_rows = [], []
    t0, n_seen = time.time(), 0

    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        prob = region_prob(model, image).cpu().numpy()
        vote = batch["consensus"].numpy() * n_raters    # consensus=V/N -> 还原票数 V
        case_ids, dias = batch["case_id"], batch["diameter_mm"].numpy()

        for b in range(prob.shape[0]):
            p = prob[b, 0].astype(np.float32)
            v = vote[b, 0]
            p_soft = v / n_raters

            gts = np.stack([(v >= k) for k in vote_levels]).astype(np.float32)   # (N,H,W)
            gt_area = gts.reshape(len(vote_levels), -1).sum(1)                    # (N,)
            preds = (p[None] > thr_np[:, None, None]).astype(np.float32)          # (T,H,W)
            pred_area = preds.reshape(len(thr_np), -1).sum(1)                     # (T,)

            inter = np.einsum("thw,ghw->tg", preds, gts)                          # (T,N)
            denom = pred_area[:, None] + gt_area[None, :]
            dice = np.where(denom > 0, 2.0 * inter / np.maximum(denom, 1e-6), 1.0)

            cid, dia = case_ids[b], float(dias[b])
            free_rows.append({
                "run": run_name, "fold": fold_tag, "case_id": cid, "diameter_mm": dia,
                # 与阈值无关的指标。注意：这两项拟合的靶 p=V/N 正是共识组 BCE 的
                # 训练目标，所以它们【不是】中立指标，只能作参考，不能当独立证据。
                "soft_dice": float(2 * (p * p_soft).sum() / max(p.sum() + p_soft.sum(), 1e-6)),
                "brier": float(np.mean((p - p_soft) ** 2)),
                "prob_mass": float(p.sum()),
                **{f"gt_area_v{k}": float(gt_area[i]) for i, k in enumerate(vote_levels)},
            })
            for ti, t in enumerate(thr_np):
                rows.append({
                    "run": run_name, "fold": fold_tag, "case_id": cid, "diameter_mm": dia,
                    # 保留 4 位小数：两端加密后有 0.001/0.005/0.998 这类值，
                    # 若按 2 位取整会被压成 0.0 / 1.0，不同阈值静默合并。
                    "threshold": round(float(t), 4),
                    **{f"dice_v{k}": float(dice[ti, i]) for i, k in enumerate(vote_levels)},
                    **{f"gt_area_v{k}": float(gt_area[i]) for i, k in enumerate(vote_levels)},
                    "area_pred": float(pred_area[ti]),
                })
            n_seen += 1

    dt = time.time() - t0
    print(f"  [{run_name}] arch={arch}  {n_seen} 样本 / {dt:.1f}s "
          f"({dt / max(n_seen, 1) * 1000:.1f} ms per sample)",
          flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows, free_rows


def write_csv(path, rows, cols, runs_written=None, overwrite=False):
    """写结果，默认【合并】而不是覆盖。

    起因：本函数原本是 open(path, "w")，直接截断重写。于是想给已有结果补一个新臂
    （例如平移臂 tgtBshift）时，只跑那一个臂就会把之前 6 个臂 x 5 折的结果全部抹掉，
    必须把 30 多个模型重评一遍才能拿回来。

    现在的语义：本次评过的 run（runs_written）在旧文件里的行被替换，
    没评的 run 原样保留。--overwrite 可强制回到旧的截断行为。

    合并会把不同时间、可能不同代码版本的结果放进同一个文件，所以每次都把
    "保留了哪些 run、重写了哪些 run" 打出来，不做静默合并。
    """
    old_rows = []
    if os.path.exists(path) and not overwrite and runs_written:
        import csv as _csv
        with open(path, encoding="utf-8", newline="") as fh:
            rd = _csv.DictReader(fh)
            same_cols = rd.fieldnames == list(cols)
            if same_cols:
                old_rows = [r for r in rd if r["run"] not in set(runs_written)]
            else:
                print(f"  [注意] {os.path.basename(path)} 的列结构与本次不同，"
                      f"无法合并，按覆盖处理")
        if same_cols:
            kept = sorted({r["run"] for r in old_rows})
            print(f"  [合并] 保留旧结果 {len(kept)} 个 run: {kept}")
            print(f"  [合并] 本次重写 {len(runs_written)} 个 run: {sorted(runs_written)}")
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(",".join(cols) + "\n")
        for r in old_rows:
            fh.write(",".join(str(r[c]) for c in cols) + "\n")
        for r in rows:
            fh.write(",".join(str(r[c]) for c in cols) + "\n")


def run_split(cfg, device, split, runs, thr, ckpt_name, out_dir, batch_size,
              overwrite=False, n_raters=4):
    split_csv = os.path.join(cfg.data.split_dir, f"{split}.csv")
    if not os.path.exists(split_csv):
        print(f"[跳过] 找不到 {split_csv}")
        return
    ds = LungNoduleDataset(cfg.data.npz_dir, split_csv, cfg, phase="val")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"\n=== split={split}  样本数={len(ds)}  阈值数={len(thr)} ===")

    vote_levels = tuple(range(1, n_raters + 1))
    all_rows, all_free, done_runs = [], [], []
    for run in runs:
        ckpt = os.path.join(cfg.outputs.root, "runs", run, "checkpoints", ckpt_name)
        if not os.path.exists(ckpt):
            print(f"  [跳过] 找不到 {ckpt}")
            continue
        r, f = eval_one(run, split, ckpt, loader, cfg, device, thr, n_raters)
        all_rows += r
        all_free += f
        done_runs.append(run)
    if not all_rows:
        return

    os.makedirs(out_dir, exist_ok=True)
    base = ["run", "fold", "case_id", "diameter_mm"]
    write_csv(os.path.join(out_dir, f"per_sample_{split}.csv"), all_rows,
              base + ["threshold"] + [f"dice_v{k}" for k in vote_levels]
              + [f"gt_area_v{k}" for k in vote_levels] + ["area_pred"],
              runs_written=done_runs, overwrite=overwrite)
    write_csv(os.path.join(out_dir, f"per_sample_free_{split}.csv"), all_free,
              base + ["soft_dice", "brier", "prob_mass"]
              + [f"gt_area_v{k}" for k in vote_levels],
              runs_written=done_runs, overwrite=overwrite)
    print(f"  -> {out_dir}\\per_sample_{split}.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--folds", default=None,
                    help="逗号分隔折号，如 0,1,2,3,4；自动配对 <prefix>_foldK 与 val_foldK")
    ap.add_argument("--prefixes", default="main,union", help="配合 --folds 使用的实验名前缀")
    ap.add_argument("--runs", default=None, help="手动模式：逗号分隔实验名（需同时给 --split）")
    ap.add_argument("--split", default=None,
                    help="划分文件名（不含 .csv）。与 --runs 搭配为手动模式；"
                         "与 --folds/--prefixes 搭配则把所有折的模型都拿到该划分上评测")
    ap.add_argument("--ckpt-name", default="best.pth")
    ap.add_argument("--thresholds", default=None,
                    help="默认 0.001~0.999（两端加密，33 个）；可手动覆盖")
    ap.add_argument("--out-dir", default="outputs/analysis")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--n-raters", type=int, default=4,
                    help="标注者数量：LIDC=4；QUBIQ 肾脏=3、脑生长=7。决定 V>=1..N 口径")
    ap.add_argument("--base", default=None,
                    help="base config（如 configs/default.yaml）与 --config 深度合并；"
                         "QUBIQ 配置只覆盖 data/loss 差异字段，必须传 default 补齐 patch_size 等")
    ap.add_argument("--overwrite", action="store_true",
                    help="截断重写输出 CSV（旧行为）。默认是合并：只替换本次评过的 run")
    args = ap.parse_args()

    cfg = load_config(args.config, args.base)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 默认阈值：两端按 logit 尺度加密到 0.001 / 0.999，中间 0.05 步长。
    # 上一轮 0.01~0.99 的扫描里，并集臂在 V>=2/3/4 上的最优卡在上界 0.99，
    # 多数投票/软靶三臂在 V>=1 上的最优卡在下界 0.01 —— 真峰值都在区间外，
    # 那些格子的"校准后"数值只是下界，必须把两端扫开才能下结论。
    default_thr = ([0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075]
                   + [round(0.10 + 0.05 * i, 3) for i in range(17)]      # 0.10 ~ 0.90
                   + [0.925, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998, 0.999])
    thr = [float(x) for x in args.thresholds.split(",")] if args.thresholds else default_thr
    print(f"device={device}  阈值={thr[0]}~{thr[-1]} 共 {len(thr)} 个")

    t0 = time.time()
    if args.folds:
        prefixes = [p.strip() for p in args.prefixes.split(",") if p.strip()]
        folds = [f.strip() for f in args.folds.split(",") if f.strip()]
        if args.split:
            # --folds + --split：把所有 <prefix>_fold<k> 都拿到同一个固定划分上评测。
            # 用途：在保留的测试集上评测全部交叉验证模型，得到无偏估计。
            runs = [f"{p}_fold{k}" for p in prefixes for k in folds]
            run_split(cfg, device, args.split, runs, thr, args.ckpt_name,
                      args.out_dir, args.batch_size, args.overwrite, args.n_raters)
        else:
            for k in folds:
                run_split(cfg, device, f"val_fold{k}", [f"{p}_fold{k}" for p in prefixes],
                          thr, args.ckpt_name, args.out_dir, args.batch_size, args.overwrite,
                          args.n_raters)
    elif args.runs and args.split:
        run_split(cfg, device, args.split, [r.strip() for r in args.runs.split(",")],
                  thr, args.ckpt_name, args.out_dir, args.batch_size, args.overwrite,
                  args.n_raters)
    else:
        raise SystemExit("需要 --folds，或同时给 --runs 和 --split")

    print(f"\n全部完成，总耗时 {time.time() - t0:.1f}s")
    print("下一步： python scripts\\analyze_symmetric.py")


if __name__ == "__main__":
    main()
