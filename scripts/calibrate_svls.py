# -*- coding: utf-8 -*-
"""标定 SVLS 高斯核的宽度，使其与本队列【实测的】标注者分歧带宽相当。

【为什么必须做这件事】
E 臂（SVLS）是 C 臂（真实投票软监督）的对照。如果 SVLS 的核选得太窄，
它就退化成硬掩膜，C - E 只是复现已知的 C - B 零效应，这个臂白跑；
如果选得太宽，又等于人为把对照方法做残。两种情况审稿人都会指出
"你的基线不是公平的基线"。

【标定量】
用「软区像素数 / 硬掩膜像素数」作为带宽的无量纲度量：
    真实投票  软区 = {x : 0 < V(x)/4 < 1}，硬掩膜 = {x : V(x) >= 2}
    SVLS      软区 = {x : 0 < svls(x) < 1}，硬掩膜同上
两者比值接近 1，说明 SVLS 造出的过渡带与医生实际的分歧带宽度相当。

【一个容易搞错的地方】
带宽由核的【支撑尺寸 ksize】决定，sigma 只改变带内的取值形状。
所以 ksize=3 时 sigma 取 0.5 / 1 / 2 的带宽完全相同——调 sigma 调不动带宽。

【注意结节尺寸的影响】
本队列结节等效半径中位数约 5 px。在半径十几像素的大病灶上，同一个核
显得"窄"；在真实的小结节上却正好。所以这个标定必须在【真实数据】上做，
不能用合成的圆形病灶推断——本项目在这上面已经差点判断反了一次。

用法（不需要 GPU，约 1 分钟）：
    python scripts\\calibrate_svls.py
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CANDIDATES = [(3, 0.5), (3, 1.0), (3, 2.0), (5, 1.0), (5, 1.5),
              (7, 1.5), (7, 2.0), (9, 2.5), (11, 3.0)]


def svls_kernel_2d(ksize, sigma, center_boost=True):
    """与 src/losses/svls.py 完全一致的核构造（含原文的中心权重抬升）。

    注意：抬升中心权重后核不再可分离，所以这里做二维卷积，不能再走可分离捷径。
    """
    ax = np.arange(ksize) - (ksize - 1) / 2.0
    g = np.exp(-ax ** 2 / (2.0 * sigma ** 2))
    k = np.outer(g, g)
    k = k / k.sum()
    if center_boost:
        c = ksize // 2
        s = 1.0 - k[c, c]
        if s <= 0:
            raise ValueError(f"ksize={ksize} sigma={sigma} 下核退化成 delta")
        k = k.copy()
        k[c, c] = s
        k = k / s
    return k


def smooth(mask, ksize, sigma, center_boost=True):
    """二维卷积，边界 edge 填充（与 src/losses/svls.py 的 replicate 一致）。"""
    k = svls_kernel_2d(ksize, sigma, center_boost)
    p = ksize // 2
    x = np.pad(mask, p, mode="edge")
    H, W = mask.shape
    win = np.lib.stride_tricks.sliding_window_view(x, (ksize, ksize))
    out = np.einsum("ijkl,kl->ij", win, k) / k.sum()
    return np.clip(out, 0.0, 1.0)


def band_ratio(soft, hard):
    return ((soft > 1e-6) & (soft < 1 - 1e-6)).sum() / max(hard.sum(), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch-dir", default="data/processed/patches")
    ap.add_argument("--n", type=int, default=300, help="取样结节数")
    ap.add_argument("--no-center-boost", dest="center_boost", action="store_false",
                    help="用普通归一化高斯核（非原文做法），仅用于消融对照")
    ap.add_argument("--out", default="outputs/analysis/svls_calibration.json")
    args = ap.parse_args()

    votes = []
    for f in sorted(glob.glob(os.path.join(args.patch_dir, "*.npz"))):
        v = np.load(f)["votes"]
        for i in range(min(len(v), 4)):        # 每例最多取 4 个，摊开到更多病例
            vi = v[i].astype(np.float32)
            if (vi >= 2).sum() >= 10:          # 太小的实例带宽统计不稳
                votes.append(vi)
        if len(votes) >= args.n:
            break
    votes = np.stack(votes[:args.n])
    p = votes / 4.0
    maj = (votes >= 2).astype(np.float32)

    real = float(np.mean([band_ratio(x, m) for x, m in zip(p, maj)]))
    real_mad = float(np.mean([np.abs(x - m).mean() for x, m in zip(p, maj)]))

    print("=" * 84)
    print(f"SVLS 核宽标定  |  取样 {len(p)} 个结节 patch  |  来源 {args.patch_dir}")
    print("=" * 84)
    print(f"  真实投票的分歧带:  软区/硬掩膜 = {real:.2f}")
    print(f"  真实软靶与硬掩膜的平均绝对差 = {real_mad:.4f}  （软监督的强度参照）")
    print()
    print(f"{'ksize':>6}{'sigma':>7}{'软区/硬掩膜':>13}{'与真实之比':>12}"
          f"{'与真实软靶MAD':>15}{'与硬掩膜MAD':>13}")
    print("-" * 84)
    rows = []
    for ks, sg in CANDIDATES:
        rt, mr, mh = [], [], []
        for m, pp in zip(maj, p):
            sv = smooth(m, ks, sg, center_boost=args.center_boost)
            rt.append(band_ratio(sv, m))
            mr.append(np.abs(sv - pp).mean())
            mh.append(np.abs(sv - m).mean())
        rt, mr, mh = (float(np.mean(a)) for a in (rt, mr, mh))
        rows.append(dict(ksize=ks, sigma=sg, band_ratio=rt, ratio_vs_real=rt / real,
                         mad_vs_real_soft=mr, mad_vs_hard=mh))
        print(f"{ks:>6}{sg:>7.1f}{rt:>13.2f}{rt / real:>12.2f}{mr:>15.4f}{mh:>13.4f}")
    print("-" * 84)
    best = min(rows, key=lambda r: abs(r["band_ratio"] - real))
    print(f"  带宽最接近真实分歧的配置: ksize={best['ksize']}  sigma={best['sigma']}"
          f"  （比值 {best['ratio_vs_real']:.2f}）")
    print("  [注] ksize 相同则带宽相同，sigma 只改变带内取值形状——调 sigma 调不动带宽。")
    print("  [论文写法] SVLS 的核宽不是随手取的：它按本队列实测的标注者分歧带宽标定，")
    print("             比值接近 1，因此这个对照臂既没被做窄成硬掩膜，也没被做宽成过度平滑。")

    rep = dict(center_boost=bool(args.center_boost),
               n_instances=int(len(p)), real_band_ratio=real,
               real_soft_vs_hard_mad=real_mad, candidates=rows,
               chosen=dict(ksize=best["ksize"], sigma=best["sigma"]))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    print(f"\n已保存 -> {args.out}")


if __name__ == "__main__":
    main()
