# -*- coding: utf-8 -*-
"""Fig. 3 重画（v12）：测试集口径 + 三分类着色（Evaluation protocol / Learning design）。

数据：outputs/analysis_test/per_sample_test.csv（158 例保留测试集）
阈值：从验证折导入（outputs/analysis_full/symmetric_analysis.json 的 table1）
口径：与 analyze_symmetric.py 的 curve 一致（先按 fold+threshold 平均，再跨 threshold 平均）
产出：outputs/figures/fig3_effect_sizes.{png,pdf}（覆盖旧图）
"""
import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

DATA = r"outputs/analysis_test/per_sample_test.csv"
THRJSON = r"outputs/analysis_full/symmetric_analysis.json"
NOISEJSON = r"outputs/analysis_full/noise_floor.json"
OUT = r"outputs/figures"
os.makedirs(OUT, exist_ok=True)

COL2 = 7.2
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.dpi": 120, "savefig.dpi": 400,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.12,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
C_BLUE, C_ORANGE = "#2a78d6", "#eb6834"
C_AQUA = "#1baf7a"
INK, INK2, INK3, GRID = "#0b0b0b", "#52514e", "#8a8983", "#e4e3de"

ARM_SHORT = {"tgtA": "A", "tgtB": "B", "tgtC": "C", "tgtD": "D", "tgtE": "E"}
LVL_SHORT = {1: r"$V\geq1$", 2: r"$V\geq2$", 3: r"$V\geq3$", 4: r"$V\geq4$"}


def curve(df, arm, k):
    s = df[(df.arm == arm) & (df[f"gt_area_v{k}"] > 0)]
    # 分析单元 = 病例：先在病例内对结节实例平均（见 docs/分析单元审计_20260904.md）
    per_case = s.groupby(["fold", "threshold", "case_id"])[f"dice_v{k}"].mean()
    return per_case.groupby(["fold", "threshold"]).mean().groupby("threshold").mean()


def curve_at(df, arm, k, t):
    c = curve(df, arm, k)
    return float(c.loc[t]) if t in c.index else None


def paired_ci(df, arm, k, t0, t1, seed=0):
    """同臂同靶、两阈值下 per-case 配对差 bootstrap CI（points）。"""
    s0 = df[(df.arm == arm) & (df[f"gt_area_v{k}"] > 0) & np.isclose(df.threshold, t0)]
    s1 = df[(df.arm == arm) & (df[f"gt_area_v{k}"] > 0) & np.isclose(df.threshold, t1)]
    g0 = s0.groupby(["fold", "case_id"])[f"dice_v{k}"].mean()
    g1 = s1.groupby(["fold", "case_id"])[f"dice_v{k}"].mean()
    g0, g1 = g0.align(g1, join="inner")
    d = (g1 - g0).values * 100
    if len(d) < 3:
        return None
    rng = np.random.default_rng(seed)
    bs = [rng.choice(d, len(d), replace=True).mean() for _ in range(2000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def main():
    sa = json.load(open(THRJSON, encoding="utf-8"))
    thr = {(a, k): sa["table1"][a][str(k)]["thr"] for a in sa["table1"] for k in range(1, 5)}
    df = pd.read_csv(DATA)
    df["arm"] = df.run.str.split("_fold").str[0]
    TGTS = [a for a in ("tgtA", "tgtB", "tgtC", "tgtD", "tgtE") if a in df.arm.unique()]

    eff = []  # (label, category, effect_in_points, ci_lo, ci_hi)
    # 1) 每臂 evaluation target spread：fixed 0.5 与导入阈值
    for a in TGTS:
        f05 = [curve_at(df, a, k, 0.5) for k in range(1, 5)]
        fimp = [curve_at(df, a, k, thr.get((a, k), 0.5)) for k in range(1, 5)]
        if all(v is not None for v in f05):
            eff.append((f"Evaluation target $V\\geq$1–4, arm {a[-1]} (fixed thr. 0.5)",
                        "Evaluation protocol", (max(f05) - min(f05)) * 100, None))
        if all(v is not None for v in fimp):
            eff.append((f"Evaluation target $V\\geq$1–4, arm {a[-1]} (imported thr.)",
                        "Evaluation protocol", (max(fimp) - min(fimp)) * 100, None))
    # 2) 每臂 @G4 阈值效应（0.5 vs 导入阈值），配对对比给 bootstrap CI
    for a in TGTS:
        v05 = curve_at(df, a, 4, 0.5)
        vim = curve_at(df, a, 4, thr.get((a, 4), 0.5))
        if v05 is not None and vim is not None and vim - v05 > 0.005:
            pci = paired_ci(df, a, 4, 0.5, thr.get((a, 4), 0.5))
            eff.append((f"Threshold, arm {a[-1]} @ {LVL_SHORT[4]} (0.5 vs imported)",
                        "Evaluation protocol", (vim - v05) * 100,
                        (pci[1], pci[2]) if pci else None))
    # 3) 监督靶范围 @G4（导入阈值）— Learning design
    vals = {a: curve_at(df, a, 4, thr.get((a, 4), 0.5)) for a in TGTS}
    if all(v is not None for v in vals.values()):
        eff.append((f"Supervision target (range over {len(TGTS)} arms) @ {LVL_SHORT[4]}",
                    "Learning design", (max(vals.values()) - min(vals.values())) * 100, None))
    # 4) 架构范围 @V2 — Learning design
    arch = [a for a in ("archPU", "archR18", "archCNX", "archPVT", "archPU1e3", "tgtB")
            if a in df.arm.unique()]
    av = {a: curve_at(df, a, 2, thr.get((a, 2), 0.5)) for a in arch}
    if all(v is not None for v in av.values()):
        eff.append((f"Architecture (range over {len(av)} models) @ {LVL_SHORT[2]}",
                    "Learning design", (max(av.values()) - min(av.values())) * 100, None))

    e = sorted(eff, key=lambda r: r[2])
    col = {"Evaluation protocol": C_BLUE, "Learning design": C_ORANGE}
    fig, ax = plt.subplots(figsize=(COL2, 0.185 * len(e) + 0.95))
    y = np.arange(len(e))
    ax.barh(y, [x[2] for x in e], height=0.66, color=[col[x[1]] for x in e], zorder=3)
    for i, x in enumerate(e):
        ax.text(x[2] + 0.2, i, f"{x[2]:.2f}", va="center", fontsize=6.4, color=INK)
        if x[3] is not None:
            lo, hi = x[3]
            ax.plot([lo, hi], [i, i], color=INK2, lw=0.8, zorder=4)
            ax.plot([lo, lo], [i - 0.18, i + 0.18], color=INK2, lw=0.8, zorder=4)
            ax.plot([hi, hi], [i - 0.18, i + 0.18], color=INK2, lw=0.8, zorder=4)
    ax.set_yticks(y, [x[0] for x in e], fontsize=6.6)
    ax.set_xlabel("Effect size (Dice points)", fontsize=7)
    ax.set_xlim(0, max(x[2] for x in e) * 1.13)

    # 噪声带（run-to-run 2σ，验证集估计）
    try:
        nf = json.load(open(NOISEJSON, encoding="utf-8"))
        noise2 = nf["fixed_0_5"]["two_sigma_pts"]
    except Exception:
        noise2 = 1.19
    ax.axvspan(0, noise2, color="#f0efea", zorder=1)

    p_max = max(x[2] for x in e if x[1] == "Evaluation protocol")
    m_max = max(x[2] for x in e if x[1] == "Learning design")
    ax.axvline(m_max, color=INK3, lw=0.8, ls=(0, (3, 2)), zorder=2)
    ax.legend(handles=[Patch(fc=col[c], label=c) for c in ("Evaluation protocol", "Learning design")],
              loc="lower right", ncol=2, bbox_to_anchor=(1.0, -0.02))
    ax.set_title(f"Evaluation-protocol choices dominate learning design  "
                 f"({p_max:.1f} vs {m_max:.1f} Dice points, {p_max / m_max:.1f}$\\times$)",
                 loc="left", pad=6)

    ax.grid(True, axis="y", color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(INK3)

    fig.savefig(os.path.join(OUT, "fig3_effect_sizes.png"), dpi=400, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(os.path.join(OUT, "fig3_effect_sizes.pdf"), bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print("-> outputs/figures/fig3_effect_sizes.png / .pdf")
    for x in e:
        ci = f"  CI[{x[3][0]:.1f},{x[3][1]:.1f}]" if x[3] else ""
        print(f"  {x[1]:<20} {x[2]:6.2f}{ci}  {x[0]}")


if __name__ == "__main__":
    main()
