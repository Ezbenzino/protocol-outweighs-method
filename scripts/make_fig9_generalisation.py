# -*- coding: utf-8 -*-
# [2026-10-03] 已被 scripts/make_figures_v22.py 取代：论文 v22 的 9 张插图统一由该脚本生成；本脚本仅作历史保留，其输出不再用于论文。
"""Fig. 9 —— 外部泛化：绝对协议效应 vs 标注者一致性指数（8 点）。

2026-09-05 重写。上一版有两处缺陷：
  1. Spearman rho 由 scipy 现算，但 p 值是**字面量 `0.048`**写死在图上的。
     0.048 是 7 点（仅 QUBIQ）的 p；8 点（含 LIDC）的精确置换 p 是 0.011。
     图上因此与 §4.14、Table 9 和自己的图注全部矛盾。
     本版把 rho 与 p 都在脚本内现算：rho 走秩的 Pearson，
     p 走 8! = 40320 的**穷举**双侧置换检验，无任何字面量、不依赖 scipy。
  2. 两个子图的标签严重重叠（(a) 右下四点堆叠，(b) 刻度与数值标签互相压字）。
     本版 (a) 改为带引线的定点标注，(b) 改为水平分组条形图。

数据来源（真实实验产物，全部现取）：
  outputs/analysis/disagreement_index.json    一致性指数 |V>=N|/|V>=1|（病例级）
  outputs/analysis/qubiq_all_effects.json     7 个 QUBIQ 任务的协议/监督效应（opt_thr）
  outputs/analysis/generalisation_table.json  LIDC 参考点

用法：
    python scripts/make_fig9_generalisation.py
"""
import itertools
import json
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "outputs/analysis"
OUT = sys.argv[2] if len(sys.argv) > 2 else "outputs/figures"
os.makedirs(OUT, exist_ok=True)

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

C_BLUE, C_ORANGE, C_VIOLET, C_GREEN = "#2a78d6", "#eb6834", "#4a3aa7", "#1a9e63"
INK, INK3, GRID = "#22221f", "#8a8983", "#e4e3de"

# (数据集键, 显示名, 副标题, 颜色, 标签锚点 (x, y), 对齐)
POINTS = [
    ("btum-t2",      "brain-tumour t2", "3 raters, MRI", C_VIOLET, (0.27, 30.0), "left"),
    ("LIDC",         "LIDC-IDRI",       "4 raters, CT",  C_BLUE,   (0.11, 24.4), "left"),
    ("brain-growth", "brain-growth",    "7 raters, MRI", C_ORANGE, (0.48, 20.2), "center"),
    ("prostate-t2",  "prostate t2",     "6 raters, MRI", C_GREEN,  (0.79, 20.4), "center"),
    ("btum-t3",      "brain-tumour t3", "3 raters, MRI", C_VIOLET, (0.44, 9.6),  "center"),
    ("btum-t1",      "brain-tumour t1", "3 raters, MRI", C_VIOLET, (0.66, 5.6),  "center"),
    ("kidney",       "kidney",          "3 raters, CT",  C_ORANGE, (0.85, 9.6),  "center"),
    ("prostate-t1",  "prostate t1",     "6 raters, MRI", C_GREEN,  (1.02, 5.0),  "right"),
]


# ---------------------------------------------------------------- 统计（纯 numpy）
def rankdata(a):
    """平均秩，处理并列。"""
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a), float)
    s = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and s[j + 1] == s[i]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return r


def spearman_rho(x, y):
    rx, ry = rankdata(x), rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    return float((rx @ ry) / np.sqrt((rx @ rx) * (ry @ ry)))


def exact_perm_p(x, y):
    """双侧精确置换 p：穷举 y 的全部 n! 个排列（n = 8 → 40320）。"""
    n = len(x)
    if n > 9:
        raise ValueError("穷举置换只在 n <= 9 时可行")
    obs = spearman_rho(x, y)
    rx = rankdata(x)
    rx = rx - rx.mean()
    denom_x = np.sqrt(rx @ rx)
    ry0 = rankdata(y)
    hit = tot = 0
    for perm in itertools.permutations(range(n)):
        ry = ry0[list(perm)]
        ry = ry - ry.mean()
        r = (rx @ ry) / (denom_x * np.sqrt(ry @ ry))
        tot += 1
        if abs(r) >= abs(obs) - 1e-12:
            hit += 1
    return obs, hit / tot, tot


# ---------------------------------------------------------------- 数据
def load_points():
    def rd(name):
        with open(os.path.join(DATA_DIR, name), encoding="utf-8") as f:
            return json.load(f)

    dis, eff, gen = (rd("disagreement_index.json"),
                     rd("qubiq_all_effects.json"),
                     rd("generalisation_table.json"))
    dis_map = {
        "brain-growth": "QUBIQ-brain", "kidney": "QUBIQ-kidney",
        "btum-t1": "QUBIQ-brain-tumor-task1", "btum-t2": "QUBIQ-brain-tumor-task2",
        "btum-t3": "QUBIQ-brain-tumor-task3", "prostate-t1": "QUBIQ-prostate-task1",
        "prostate-t2": "QUBIQ-prostate-task2"}
    eff_map = dict(dis_map, **{"brain-growth": "QUBIQ-brain-growth"})

    out = {"LIDC": dict(agree=gen["agreement_index"]["LIDC"],
                        protocol=gen["LIDC"]["opt_thr"]["protocol"],
                        supervision=gen["LIDC"]["opt_thr"]["supervision"],
                        ratio=gen["LIDC"]["opt_thr"]["ratio"])}
    for short, key in eff_map.items():
        o = eff[key]["effects"]["opt_thr"]
        out[short] = dict(agree=dis[dis_map[short]]["area_mean_ratio"],
                          protocol=o["protocol"], supervision=o["supervision"],
                          ratio=o["ratio"])
    return out


def main():
    pts = load_points()
    keys = [p[0] for p in POINTS]
    xs = np.array([pts[k]["agree"] for k in keys])
    proto = np.array([pts[k]["protocol"] for k in keys])
    superv = np.array([pts[k]["supervision"] for k in keys])

    rho, pval, n_perm = exact_perm_p(xs, proto)

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.6, 3.35),
                                   gridspec_kw={"width_ratios": [1.0, 0.92]})

    # ---- (a) 绝对协议效应 vs 一致性指数 --------------------------------
    for i, (k, name, sub, c, anchor, ha) in enumerate(POINTS):
        axa.scatter(xs[i], proto[i], color=c, s=40, zorder=4,
                    edgecolor="white", linewidth=0.6)
        axa.annotate(f"{name}\n{sub}", xy=(xs[i], proto[i]), xytext=anchor,
                     textcoords="data", fontsize=6.9, color=c, ha=ha,
                     va="center", linespacing=1.25, zorder=5,
                     arrowprops=dict(arrowstyle="-", color=c, lw=0.55,
                                     shrinkA=1.5, shrinkB=4.0, alpha=0.75))
    axa.set_xlabel("Rater agreement index  $|V\\geq N|\\,/\\,|V\\geq 1|$\n"
                   "(larger = raters agree more)")
    axa.set_ylabel("Protocol effect (Dice points, optimal threshold)")
    axa.set_xlim(0.06, 1.10)
    axa.set_ylim(-1.5, 35.5)
    axa.text(0.98, 0.985,
             f"Spearman $\\rho$ = {rho:.2f}\n"
             f"exact permutation $p$ = {pval:.3f}",
             transform=axa.transAxes, fontsize=7.2, ha="right", va="top",
             color="#52514e")
    axa.set_title("a", loc="left", fontweight="bold")

    # ---- (b) 两分量效应量，水平条形，按一致性升序 ----------------------
    order = np.argsort(xs)[::-1]           # 顶部 = 一致性最低
    labels = [f"{POINTS[i][1]}  ({xs[i]:.2f})" for i in order]
    y = np.arange(len(order))
    h = 0.36
    axb.barh(y - h / 2, proto[order], h, color=C_BLUE,
             label="Evaluation target (protocol)", zorder=3)
    axb.barh(y + h / 2, superv[order], h, color=C_ORANGE,
             label="Supervision target (method)", zorder=3)
    for yy, v in zip(y - h / 2, proto[order]):
        axb.text(v + 0.6, yy, f"{v:.1f}", va="center", ha="left",
                 fontsize=6.4, fontweight="bold", color=INK, zorder=5)
    for yy, v in zip(y + h / 2, superv[order]):
        axb.text(v + 0.6, yy, f"{v:.1f}", va="center", ha="left",
                 fontsize=6.4, fontweight="bold", color=INK, zorder=5)
    axb.set_yticks(y)
    axb.set_yticklabels(labels, fontsize=6.8)
    axb.invert_yaxis()
    axb.set_xlim(0, 36)
    axb.set_ylim(len(order) - 0.55, -0.55)
    axb.set_xlabel("Effect size (Dice points)")
    axb.tick_params(axis="y", length=0)
    axb.set_ylabel("Task  (rater agreement index)", fontsize=7.4)
    axb.legend(loc="upper right", bbox_to_anchor=(1.0, 0.995), fontsize=6.8,
               handlelength=1.0, handleheight=1.0, borderpad=0.2,
               labelspacing=0.3)
    axb.set_title("b", loc="left", fontweight="bold")

    axa.grid(True, axis="y", color=GRID, lw=0.5, zorder=0)
    axb.grid(True, axis="x", color=GRID, lw=0.5, zorder=0)
    for ax in (axa, axb):
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_color(INK3)
    axb.spines["left"].set_visible(False)

    fig.tight_layout(w_pad=1.6)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig9_generalisation.{ext}"))
    plt.close(fig)

    print(f"  -> {OUT}/fig9_generalisation.png / .pdf")
    print(f"  Spearman rho = {rho:.4f}, exact permutation p = {pval:.4f} "
          f"({n_perm} permutations)")
    for i, k in enumerate(keys):
        print(f"  校验: {k:14s} agree={xs[i]:.4f} protocol={proto[i]:6.2f} "
              f"supervision={superv[i]:5.2f}")
    # 与论文口径的硬校验：图上任何数字都必须与 Table 9 / §4.14 一致
    assert abs(rho - (-0.8571)) < 5e-4, f"rho 与论文 -0.86 不符: {rho}"
    assert abs(pval - 0.0107) < 5e-4, f"p 与论文 0.011 不符: {pval}"
    print("  自检通过：rho 与 p 与正文一致")


if __name__ == "__main__":
    main()
