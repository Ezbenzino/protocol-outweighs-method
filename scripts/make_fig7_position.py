# [2026-10-03] 已被 scripts/make_figures_v22.py 取代：论文 v22 的 9 张插图统一由该脚本生成；本脚本仅作历史保留，其输出不再用于论文。
"""Fig. 7 — 位置先验曲线：评测窗偏移 vs Dice（对 G2）。

数据来源：outputs/analysis_scope/diag_offset_allarms.json
  由 scripts/diag_offset.py 生成（评测窗相对病灶中心平移 0..56 px，
  各臂在 per-arm 校准阈值下对 V>=2 多数票的 Dice，逐结节均值）。

主臂：tgtB（中心裁剪，紫色）与 tgtBshift（±32 px 平移增强，橙色）；
其余 9 个未做平移增强的臂以灰色细线作背景。

用法：
    python scripts/make_fig7_position.py
"""
import json
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA = sys.argv[1] if len(sys.argv) > 1 else "outputs/analysis_scope/diag_offset_allarms.json"
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

C_ORANGE, C_VIOLET = "#eb6834", "#4a3aa7"
INK3, GRID = "#8a8983", "#e4e3de"

MAIN, SHIFT = "tgtB_fold0", "tgtBshift_fold0"
AUG_RANGE = 32  # tgtBshift 训练时平移增强幅度（px）


def main():
    with open(DATA, encoding="utf-8") as f:
        d = json.load(f)
    offsets = [int(o) for o in d["offsets"]]
    runs = d["runs"]
    assert MAIN in runs and SHIFT in runs, f"{DATA} 缺 {MAIN} / {SHIFT}"

    fig, ax = plt.subplots(figsize=(7.2, 3.4))

    # 背景：未做平移增强的其余臂
    for name, r in runs.items():
        if name in (MAIN, SHIFT):
            continue
        ys = [r["mean_by_offset"][str(o)] for o in offsets]
        ax.plot(offsets, ys, color=INK3, lw=1.0, alpha=0.55, zorder=2)

    # 主臂
    yb = [runs[MAIN]["mean_by_offset"][str(o)] for o in offsets]
    ys = [runs[SHIFT]["mean_by_offset"][str(o)] for o in offsets]
    ax.plot(offsets, yb, color=C_VIOLET, lw=2.2, marker="o", ms=4,
            label="B majority (centre-cropped)", zorder=4)
    ax.plot(offsets, ys, color=C_ORANGE, lw=2.2, marker="s", ms=4,
            label="+ translation augmentation ($\\pm$32 px)", zorder=4)
    ax.plot([], [], color=INK3, lw=1.0, alpha=0.55,
            label=f"{len(runs) - 2} other arms")

    # 增强幅度阴影（先于曲线绘制会被遮挡，这里只画边界提示用的浅色带）
    ax.axvspan(0, AUG_RANGE, color="#f0efea", zorder=0)
    ax.text(AUG_RANGE / 2, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0.95,
            "", ha="center")  # 占位，真正标注在 ylim 设定后补

    ax.set_xlabel("Offset of the evaluation window from the lesion centre (px)")
    ax.set_ylabel("Dice against $G_2$")
    ax.set_xticks(offsets)
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(INK3)

    ymax = ax.get_ylim()[1]
    ax.text(AUG_RANGE / 2, ymax * 0.97, "training augmentation range",
            ha="center", va="top", fontsize=7.5, color="#52514e",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
    ax.annotate("8 px $\\approx$ 1.6 lesion radii", xy=(8, yb[1]),
                xytext=(20, yb[1] + 0.02), fontsize=7.5, color="#52514e",
                arrowprops=dict(arrowstyle="-", color="#52514e", lw=0.8),
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig7_position_prior.{ext}"))
    plt.close(fig)
    print(f"  -> {OUT}/fig7_position_prior.png / .pdf")
    print(f"  校验: tgtB d0={yb[0]:.4f} d8={yb[1]:.4f} | tgtBshift d32={ys[4]:.4f}")


if __name__ == "__main__":
    main()
