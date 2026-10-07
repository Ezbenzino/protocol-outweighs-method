# -*- coding: utf-8 -*-
"""评测空间（evaluation scope）作为受控因素 —— 论文的第三个协议因素 F3。

**这个脚本回答的问题**：论文此前所有 Dice 都是在以结节为中心的 128px patch 上算的，
等于假设检测器完美（oracle localisation），且肺内其他位置的假阳性永远不被计入。
文献里 2D 切片级、3D patch、整体积三种口径并存，但几乎没人量化过换口径要付多少代价。

本脚本把"换口径"拆成三段，每段只动一个变量：

  patch128  模型输入 = 以结节为中心的 128 裁剪（零填充），评测 = 同一块 128
            —— 完全复刻现有协议。它应当【精确复现】eval_symmetric.py 的结果，
               是整条流水线的端到端自检。
  win128    模型输入 = 完整 512 切片，评测 = 以同一结节为中心的 128 窗口
            —— 与上一行只差"模型看到多少上下文"。差值 = 输入上下文效应。
  win192 / win256 / win384 / win512
            模型输入不变（完整切片），只把评测窗口逐级放大。
            窗口每放大一圈，进入统计的只有更多背景像素，因此 Dice 的下降
            【完全归因于更大视野里累积的假阳性】，不掺杂任何其他变化。
            这才是干净的"评测空间效应"。

  slice     完整 512x512，仅含结节的切片，逐切片求 Dice 再按病例平均
  volume    完整 512x512，全部切片，按病例聚合成一个 Dice
            —— 这两行附带报告，但解释受限：本研究的模型只在"以结节为中心的
               patch"上训练过，从未见过不含结节的区域，它在这两个口径上的
               表现同时受"评测空间"和"训练分布"影响，不能单独归因为评测空间
               效应。正文必须写明这一点。

【为什么窗口用裁剪而不是零填充】
靠近图像边缘时窗口会超出 512。超出部分若按零填充处理：预测概率 0，在任何
大于 0 的阈值下都不会被判为阳性；GT 票数 0，也不属于任何层级。两者都不进入
Dice 的分子分母，所以直接裁掉与零填充【结果完全一致】，只是省了内存。

【为什么不逐阈值物化掩膜】
朴素做法是对 T 个阈值各生成一张二值图再和 4 个 GT 求交，代价 O(T x 4 x H x W)。
33 阈值 x 4 层级 x 262144 像素 x 3.5 万切片 ~ 1.2 万亿次操作，不可行。
本脚本改用精确等价的计数法：把每个像素的概率用 searchsorted 映射到"它超过了
前几个阈值"，再用 bincount 做直方图，反向累加即得所有阈值下的计数。
代价降到 O(H x W log T)，且结果与逐阈值物化【完全一致】，不是近似。

用法（PowerShell，在仓库根目录下）：
    # 先用 8 个病例测吞吐，估算总时间
    python scripts\\eval_scope.py --runs tgtB_fold0,tgtA_fold0 --split test --max-cases 8
    # 确认后跑全量
    python scripts\\eval_scope.py --runs tgtB_fold0,tgtA_fold0 --split test
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.models.from_ckpt import load_for_eval, region_prob
from src.data.utils import roi_center_crop
from src.utils.config import load_config

VOTE_LEVELS = (1, 2, 3, 4)
WINDOWS = (128, 192, 256, 384, 512)   # 视野阶梯；128 对应已有的 patch 级评测范围
PATCH = 128                            # 训练/现有评测所用的 patch 边长


def counts_all_thresholds(prob, vote, thr):
    """一次算出所有阈值下的 (预测面积, 与各 GT 层级的交集面积)。

    prob : (H,W) float32 预测概率
    vote : (H,W) 票数 0..4
    thr  : (T,) 升序阈值
    返回 pred_area (T,), inter (T,4), gt_area (4,)

    原理：idx = searchsorted(thr, p, 'left') 等于"p 严格大于的阈值个数"，
    因此 p > thr[j] 等价于 idx > j。对 idx 做直方图再反向累加，
    就一次性得到所有 j 的计数——与逐阈值二值化的结果逐位相等。
    """
    T = len(thr)
    p = np.ascontiguousarray(prob).ravel()
    v = np.ascontiguousarray(vote).ravel()
    idx = np.searchsorted(thr, p, side="left")          # 0..T
    tail = lambda c: np.cumsum(c[::-1])[::-1][1:]       # tail[j] = sum(c[j+1:])

    pred_area = tail(np.bincount(idx, minlength=T + 1)).astype(np.float64)
    inter = np.empty((T, 4), np.float64)
    gt_area = np.empty(4, np.float64)
    for i, k in enumerate(VOTE_LEVELS):
        m = v >= k
        gt_area[i] = m.sum()
        inter[:, i] = tail(np.bincount(idx[m], minlength=T + 1)) if gt_area[i] else 0.0
    return pred_area, inter, gt_area


def accumulate_dice(acc, n, pa, it, ga):
    """把一个评测单元的全阈值 Dice 加进累加器；GT 为空的层级跳过（Dice 无定义）。"""
    den = pa[:, None] + ga[None, :]
    d = np.where(den > 0, 2.0 * it / np.maximum(den, 1e-9), 1.0)
    ok = ga > 0
    acc[:, ok] += d[:, ok]
    n[ok] += 1


@torch.no_grad()
def eval_case(model, device, npz_path, meta_path, thr, batch, in_ch):
    """一个病例：完整切片推理（slice/volume/win 阶梯）+ patch 推理（patch128 自检）。"""
    z = np.load(npz_path, mmap_mode="r")
    sl, vt = z["slices"], z["consensus"]
    n = sl.shape[0]
    with open(meta_path, encoding="utf-8") as f:
        nods = json.load(f).get("nodules", [])
    nod_slices = sorted({int(x["slice"]) for x in nods})
    nod_set = set(nod_slices)
    # 每个含结节的切片上，该切片里所有结节的中心，用于裁同心评测窗口
    centres = {}
    for x in nods:
        centres.setdefault(int(x["slice"]), []).append((float(x["cy"]), float(x["cx"])))

    T = len(thr)
    acc = {f"win{w}": np.zeros((T, 4), np.float64) for w in WINDOWS}
    cnt = {f"win{w}": np.zeros(4, np.int64) for w in WINDOWS}
    acc["patch128"] = np.zeros((T, 4), np.float64)
    cnt["patch128"] = np.zeros(4, np.int64)
    acc["slice"] = np.zeros((T, 4), np.float64)
    cnt["slice"] = np.zeros(4, np.int64)
    v_pred = np.zeros(T, np.float64)
    v_inter = np.zeros((T, 4), np.float64)
    v_gt = np.zeros(4, np.float64)

    # ---------- 第一遍：完整 512 切片推理 ----------
    for a in range(0, n, batch):
        b = min(a + batch, n)
        img = sl[a:b].astype(np.float32) / 255.0
        x = torch.from_numpy(img)[:, None]
        if in_ch == 3:
            x = x.repeat(1, 3, 1, 1)
        prob = region_prob(model, x.to(device)).float().cpu().numpy()[:, 0]
        for j in range(b - a):
            si = a + j
            vs = vt[si].astype(np.int16)
            pa, it, ga = counts_all_thresholds(prob[j], vs, thr)
            v_pred += pa
            v_inter += it
            v_gt += ga
            if si not in nod_set:
                continue
            accumulate_dice(acc["slice"], cnt["slice"], pa, it, ga)
            # 视野阶梯：同一模型、同一切片、同一中心，只放大评测窗口
            H, Wd = prob[j].shape
            for (cy, cx) in centres.get(si, []):
                iy, ix = int(round(cy)), int(round(cx))
                for w in WINDOWS:
                    h = w // 2
                    y0, y1 = max(iy - h, 0), min(iy + h, H)
                    x0, x1 = max(ix - h, 0), min(ix + h, Wd)
                    pa2, it2, ga2 = counts_all_thresholds(
                        prob[j][y0:y1, x0:x1], vs[y0:y1, x0:x1], thr)
                    accumulate_dice(acc[f"win{w}"], cnt[f"win{w}"], pa2, it2, ga2)

    # ---------- 第二遍：patch 推理，复刻现有协议 ----------
    for a in range(0, len(nods), batch):
        chunk = nods[a:a + batch]
        imgs, votes = [], []
        for x in chunk:
            si = int(x["slice"])
            c = (float(x["cy"]), float(x["cx"]))
            imgs.append(roi_center_crop(sl[si].astype(np.float32) / 255.0, c, PATCH))
            votes.append(roi_center_crop(vt[si].astype(np.int16), c, PATCH))
        xb = torch.from_numpy(np.stack(imgs))[:, None]
        if in_ch == 3:
            xb = xb.repeat(1, 3, 1, 1)
        pb = region_prob(model, xb.to(device)).float().cpu().numpy()[:, 0]
        for j in range(len(chunk)):
            pa, it, ga = counts_all_thresholds(pb[j], votes[j], thr)
            accumulate_dice(acc["patch128"], cnt["patch128"], pa, it, ga)

    return acc, cnt, v_pred, v_inter, v_gt, n, len(nod_slices), len(nods)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--runs", required=True, help="逗号分隔实验名，如 tgtB_fold0,tgtA_fold0")
    ap.add_argument("--split", default="test")
    ap.add_argument("--ckpt-name", default="best.pth")
    ap.add_argument("--thresholds", default=None)
    ap.add_argument("--max-cases", type=int, default=0, help=">0 时只跑前 N 个病例，用于测吞吐")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out-dir", default="outputs/analysis_scope")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    thr = np.asarray(sorted(float(x) for x in args.thresholds.split(",")) if args.thresholds
                     else ([0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075]
                           + [round(0.10 + 0.05 * i, 3) for i in range(17)]
                           + [0.925, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998, 0.999]), np.float32)

    split_csv = os.path.join(cfg.data.split_dir, f"{args.split}.csv")
    with open(split_csv, encoding="utf-8") as f:
        cases = [ln.strip() for ln in f if ln.strip() and ln.strip() != "case_id"]
    if args.max_cases:
        cases = cases[:args.max_cases]
    scopes = ["patch128"] + [f"win{w}" for w in WINDOWS] + ["slice"]
    print(f"device={device}  split={args.split}  病例 {len(cases)}  阈值 {len(thr)}")
    print(f"评测空间: {scopes} + volume")

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"scope_{args.split}.csv")
    cols = ["run", "case_id", "scope", "threshold", "level", "dice", "n_units",
            "n_slices", "n_nodule_slices", "n_nodules"]
    fh = open(out, "w", encoding="utf-8", newline="")
    fh.write(",".join(cols) + "\n")

    for run in [r.strip() for r in args.runs.split(",") if r.strip()]:
        ck = os.path.join(cfg.outputs.root, "runs", run, "checkpoints", args.ckpt_name)
        if not os.path.exists(ck):
            print(f"  [跳过] 找不到 {ck}")
            continue
        model, arch, act = load_for_eval(ck, cfg)
        model.to(device).eval()
        print(f"\n[{run}] arch={arch}")

        t0, done, tot_sl = time.time(), 0, 0
        for ci, c in enumerate(cases):
            npz = os.path.join(cfg.data.npz_dir, f"{c}.npz")
            meta = os.path.join(cfg.data.npz_dir, f"{c}.json")
            if not (os.path.exists(npz) and os.path.exists(meta)):
                continue
            acc, cnt, vp, vi, vg, nsl, nnod_sl, nnod = eval_case(
                model, device, npz, meta, thr, args.batch, cfg.model.in_channels)
            done += 1
            tot_sl += nsl
            for li, k in enumerate(VOTE_LEVELS):
                for ti, t in enumerate(thr):
                    for sc in scopes:
                        if cnt[sc][li]:
                            fh.write(f"{run},{c},{sc},{t:.4f},{k},"
                                     f"{acc[sc][ti, li] / cnt[sc][li]:.6f},{cnt[sc][li]},"
                                     f"{nsl},{nnod_sl},{nnod}\n")
                    den = vp[ti] + vg[li]
                    dv = 2.0 * vi[ti, li] / den if den > 0 else 1.0
                    fh.write(f"{run},{c},volume,{t:.4f},{k},{dv:.6f},1,"
                             f"{nsl},{nnod_sl},{nnod}\n")
            if done % 10 == 0 or ci == len(cases) - 1:
                el = time.time() - t0
                rate = tot_sl / max(el, 1e-9)
                eta = (len(cases) - done) * (el / max(done, 1)) / 60
                print(f"  {done}/{len(cases)} 例  {tot_sl} 切片  {el/60:.1f} 分钟  "
                      f"{rate:.1f} 切片/秒  剩余约 {eta:.1f} 分钟", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    fh.close()
    print(f"\n已保存 -> {out}")
    print("下一步： python scripts\\analyze_scope.py")


if __name__ == "__main__":
    main()
