# -*- coding: utf-8 -*-
# [2026-10-03] 已被 scripts/make_figures_v22.py 取代：论文 v22 的 9 张插图统一由该脚本生成；本脚本仅作历史保留，其输出不再用于论文。
"""生成论文插图 v2 —— 支持任意多个实验臂。全部数据来自真实实验，无任何编造数值。

v2 相对 v1：从"consensus / union 两个臂"改成"按 run 名自动识别的 N 个臂"，
以承载四个训练靶（tgtA/B/C/D）与后续的架构对照臂。

数据来源：
  outputs/analysis_full/per_sample_val_fold{0..4}.csv        逐样本 x 逐阈值 x 4 个评测靶的 Dice
  outputs/analysis_full/per_sample_free_val_fold{0..4}.csv   逐样本的与阈值无关指标与 GT 面积
  data/processed/patches/LIDC-IDRI-0452.npz             Fig.1 的真实 CT patch 与投票图

配色取自 dataviz skill 的验证过的分类色板（validate_palette.js 四色全项通过：
亮度带、色度下限、CVD 相邻对分离 ΔE 9.1、常视觉分离 ΔE 22.9）。
对比度 WARN 项由论文场景自动满足：所有图都有直接标注且正文附对应数表。
"""
import glob
import json
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy import stats

# 默认改为 analysis_full：outputs/analysis 是 2026-08-31 被覆盖后的残缺副本（仅 8 个 run）
DATA = sys.argv[1] if len(sys.argv) > 1 else "outputs/analysis_full"
PATCH = sys.argv[2] if len(sys.argv) > 2 else "data/processed/patches/LIDC-IDRI-0452.npz"
OUT = sys.argv[3] if len(sys.argv) > 3 else "outputs/figures"
os.makedirs(OUT, exist_ok=True)

COL1, COL2 = 3.5, 7.2                                  # 单栏 / 双栏宽度 (inch)
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

C_BLUE, C_ORANGE, C_AQUA, C_YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
C_MAGENTA, C_VIOLET = "#e87ba4", "#4a3aa7"
INK, INK2, INK3, GRID = "#0b0b0b", "#52514e", "#8a8983", "#e4e3de"

ARM_LABEL = {
    "tgtA": "A  Union ($V\\geq1$)",
    "tgtB": "B  Majority ($V\\geq2$, hard)",
    "tgtC": "C  Consensus ($V\\geq2$ + soft $p$)",
    "tgtD": "D  Soft only ($p=V/4$)",
    "archR18": "ResNet-18 U-Net",
    "archPU": "Plain U-Net",
    "main": "[legacy] 6-loss consensus",
    "union": "[legacy] union",
}
ARM_SHORT = {"tgtA": "A Union", "tgtB": "B Majority", "tgtC": "C Consensus", "tgtD": "D Soft",
             "archR18": "ResNet-18", "archPU": "Plain U-Net",
             "main": "[legacy] cons.", "union": "[legacy] union"}
ARM_COLOR = {"tgtA": C_ORANGE, "tgtB": C_AQUA, "tgtC": C_BLUE, "tgtD": C_YELLOW,
             "archR18": C_MAGENTA, "archPU": C_VIOLET, "main": C_BLUE, "union": C_ORANGE}
ARM_ORDER = ["tgtA", "tgtB", "tgtC", "tgtD", "archR18", "archPU", "main", "union"]

# 森林图展示的关键对比：(臂1, 臂2, 这一对回答什么)
CONTRASTS = [("tgtC", "tgtA", "C $-$ A  (consensus vs union)"),
             ("tgtB", "tgtA", "B $-$ A  (mask extent only)"),
             ("tgtC", "tgtB", "C $-$ B  (soft supervision only)")]

LEVELS = (1, 2, 3, 4)
LVL_TITLE = {1: r"(a)  vs $V\geq1$ (union)", 2: r"(b)  vs $V\geq2$ (majority)",
             3: r"(c)  vs $V\geq3$", 4: r"(d)  vs $V\geq4$ (unanimous)"}
LVL_SHORT = {1: r"$V\geq1$", 2: r"$V\geq2$", 3: r"$V\geq3$", 4: r"$V\geq4$"}


def style(ax, grid_axis="y"):
    ax.grid(True, axis=grid_axis, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(INK3)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  -> {name}.png / .pdf")


# ---------------------------------------------------------------- 载入
def load():
    fs = [f for f in sorted(glob.glob(os.path.join(DATA, "per_sample_val_fold*.csv")))
          if "free" not in os.path.basename(f)]
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    df["arm"] = df.run.str.split("_fold").str[0]
    ff = sorted(glob.glob(os.path.join(DATA, "per_sample_free_val_fold*.csv")))
    free = pd.concat([pd.read_csv(f) for f in ff], ignore_index=True)
    free["arm"] = free.run.str.split("_fold").str[0]
    return df, free


def sub(df, arm, k, thr=None):
    """某臂、某评测靶下 GT 非空的样本（GT 为空时 Dice 无定义）。"""
    m = (df.arm == arm) & (df[f"gt_area_v{k}"] > 0)
    if thr is not None:
        m &= np.isclose(df.threshold, thr)
    return df[m]


def curve(df, arm, k):
    """阈值-Dice 曲线：先按折平均再跨折平均，避免样本多的折主导。"""
    s = sub(df, arm, k)
    per_case = s.groupby(["fold", "threshold", "case_id"])[f"dice_v{k}"].mean()
    return per_case.groupby(["fold", "threshold"]).mean().groupby("threshold").mean()


df, free = load()
present = list(df.arm.unique())
ARMS = [a for a in ARM_ORDER if a in present] + [a for a in present if a not in ARM_ORDER]
TGTS = [a for a in ARMS if a.startswith("tgt")] or ARMS
CURVES = {(a, k): curve(df, a, k) for a in ARMS for k in LEVELS}
BEST = {(a, k): (float(c.idxmax()), float(c.max()), float(c.loc[0.5])) for (a, k), c in CURVES.items()}
FOLDS = sorted(df.fold.unique())
N_CASES = int(df.groupby("fold").case_id.nunique().sum())
lab = lambda a: ARM_LABEL.get(a, a)
print(f"载入 {len(FOLDS)} 折 / {N_CASES} 病例 / {len(sub(df, ARMS[0], 1, 0.5))} 样本 / "
      f"{df.threshold.nunique()} 阈值 / 臂: {', '.join(ARMS)}")


# ------------------------------------------------ 噪声地板（种子重复实验，analysis_full 全量 fixed-0.5 口径）
def seed_noise_floor(seed_dir="outputs/analysis_full"):
    """从 analysis_full 全量（2 臂 x 2 折 x 3 种子）估计单次训练的随机波动。

    口径与论文 §4.8 一致：固定阈值 0.5 下逐 (arm,fold,target) 组 3-seed 组内
    标准差合并 -> pooled sigma (32 dof)。论文报告 2sigma = 1.15 点。
    返回 (合并SD, 参与估计的臂数, 每臂种子数)；目录不存在时返回 (None, 0, 0)。
    """
    fs = sorted(glob.glob(os.path.join(seed_dir, "per_sample_val_fold*.csv")))
    if not fs:
        return None, 0, 0
    sd_df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    sd_df["base"] = (sd_df.run.str.replace(r"^seed_", "", regex=True)
                     .str.replace(r"_fold\d+", "", regex=True)
                     .str.replace(r"_s\d+$", "", regex=True))
    sd_df["foldn"] = sd_df.fold.str.replace("val_fold", "")
    devs, n_arms, n_seeds = [], 0, 0
    for base, g in sd_df.groupby("base"):
        if base not in ("tgtA", "tgtB"):
            continue
        g = g[g.run.str.contains(r"_s\d+$", regex=True)]   # 只保留 3 个种子 run
        for foldn in ("0", "1"):
            gf = g[g.foldn == foldn]
            runs = sorted(gf.run.unique())
            if len(runs) < 2:
                continue
            n_arms += 1
            n_seeds = max(n_seeds, len(runs))
            for k in LEVELS:
                v = np.array([gf[(gf.run == r) & (gf[f"gt_area_v{k}"] > 0)
                                & (np.isclose(gf.threshold, 0.5, atol=1e-6))][f"dice_v{k}"].mean()
                              for r in runs])
                devs.extend(v - v.mean())
    if not devs:
        return None, 0, 0
    n_groups = n_arms * len(LEVELS)
    dof = len(devs) - n_groups
    return float(np.sqrt(np.sum(np.asarray(devs) ** 2) / dof)), n_arms, n_seeds


NOISE_SD, NOISE_ARMS, NOISE_SEEDS = seed_noise_floor(
    os.path.join(os.path.dirname(DATA.rstrip("/\\")), "analysis_full"))
if NOISE_SD:
    print(f"噪声地板：{NOISE_ARMS} 个臂 x {NOISE_SEEDS} 个种子，合并 SD = {NOISE_SD*100:.2f} 点")


# ================================================================ Fig. 1
def fig1():
    """标注共识层级的构成：真实 CT patch + 投票图 + 四个嵌套 GT。

    128px patch 里结节只占很小一块，直接画看不清，所以按投票图的非零区域
    自动裁一个正方形视窗（1.9 倍余量）。
    """
    z = np.load(PATCH)
    img, vote = z["images"][3], z["votes"][3]      # nodule3: 四个层级都非空
    ys, xs = np.nonzero(vote)
    cy, cx = int(ys.mean()), int(xs.mean())
    r = int(max(ys.max() - ys.min(), xs.max() - xs.min()) * 1.9 / 2) + 4
    y0, y1 = max(cy - r, 0), min(cy + r, vote.shape[0])
    x0, x1 = max(cx - r, 0), min(cx + r, vote.shape[1])
    img_c, vote_c = img[y0:y1, x0:x1], vote[y0:y1, x0:x1]

    fig, axes = plt.subplots(1, 6, figsize=(COL2, 1.62))
    fig.subplots_adjust(wspace=0.08)
    tag = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    ttl = ["CT patch", "Vote map", r"$V\geq1$", r"$V\geq2$", r"$V\geq3$", r"$V\geq4$"]
    for i, ax in enumerate(axes):
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color(INK3); sp.set_linewidth(0.5)
        ax.set_title(f"{tag[i]} {ttl[i]}", pad=3, fontsize=7.5)
        if i == 0:
            ax.imshow(img_c, cmap="gray", vmin=0, vmax=255)
        elif i == 1:
            im = ax.imshow(vote_c, cmap="magma", vmin=0, vmax=4, interpolation="nearest")
        else:
            k = i - 1
            ax.imshow(img_c, cmap="gray", vmin=0, vmax=255, alpha=0.9)
            m = np.ma.masked_where(vote_c < k, np.ones_like(vote_c))
            ax.imshow(m, cmap=mpl.colors.ListedColormap([C_BLUE]), alpha=0.45, interpolation="nearest")
            ax.contour(vote_c >= k, levels=[0.5], colors=[C_BLUE], linewidths=1.0)
            ax.set_xlabel(f"{int((vote >= k).sum())} px", fontsize=6.8, color=INK2, labelpad=2)
    cax = fig.add_axes([0.905, 0.30, 0.008, 0.42])
    cb = fig.colorbar(im, cax=cax, ticks=[0, 1, 2, 3, 4])
    cb.ax.tick_params(labelsize=6, width=0.5, length=2)
    cb.outline.set_linewidth(0.5)
    cb.set_label("raters", fontsize=6.5, labelpad=2)
    save(fig, "fig1_consensus_levels")


# ================================================================ Fig. 2
def fig2():
    """阈值-Dice 曲线：全文最关键的一张图，展示固定 0.5 的比较为何无效。"""
    n = len(TGTS)
    fig, axes = plt.subplots(1, 4, figsize=(COL2, 2.45), sharey=True)
    fig.subplots_adjust(bottom=0.40)
    for ax, k in zip(axes, LEVELS):
        for a in TGTS:
            c = CURVES[(a, k)]
            t_opt, v_opt, v_05 = BEST[(a, k)]
            ax.plot(c.index, c.values, color=ARM_COLOR.get(a, INK2), lw=1.5, zorder=3)
            ax.plot([t_opt], [v_opt], "o", ms=4.6, color=ARM_COLOR.get(a, INK2),
                    mec="white", mew=0.9, zorder=5)
            ax.plot([0.5], [v_05], "s", ms=4.2, color="white",
                    mec=ARM_COLOR.get(a, INK2), mew=1.2, zorder=5)
        ax.axvline(0.5, color=INK3, lw=0.7, ls=(0, (3, 2)), zorder=1)
        ax.set_title(LVL_TITLE[k], pad=4, loc="left")
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0", "", "0.5", "", "1"])
        ax.set_xlabel("Threshold", labelpad=1)
        style(ax, "both")
        spread05 = max(BEST[(a, k)][2] for a in TGTS) - min(BEST[(a, k)][2] for a in TGTS)
        spreadop = max(BEST[(a, k)][1] for a in TGTS) - min(BEST[(a, k)][1] for a in TGTS)
        ax.text(0.03, 0.04, f"spread$_{{0.5}}$ = {spread05*100:.2f}\n"
                            f"spread$_{{opt}}$ = {spreadop*100:.2f}",
                transform=ax.transAxes, fontsize=6.4, color=INK, va="bottom",
                bbox=dict(fc="white", ec=GRID, lw=0.5, pad=1.8))
    axes[0].set_ylabel("Dice")
    lo = min(BEST[(a, k)][2] for a in TGTS for k in LEVELS)
    axes[0].set_ylim(max(0.3, lo - 0.08), 0.92)
    h = [plt.Line2D([], [], color=ARM_COLOR.get(a, INK2), lw=1.5, label=lab(a)) for a in TGTS]
    h += [plt.Line2D([], [], ls="none", marker="o", ms=4.6, color=INK3, mec="white",
                     label="per-target optimum"),
          plt.Line2D([], [], ls="none", marker="s", ms=4.2, color="white", mec=INK3, mew=1.2,
                     label="fixed threshold 0.5")]
    fig.legend(handles=h, loc="lower center", ncol=min(4, n + 1), bbox_to_anchor=(0.5, -0.02),
               fontsize=6.4, labelspacing=0.5, columnspacing=1.4, handlelength=1.2)
    save(fig, "fig2_threshold_curves")


# ================================================================ Fig. 3
def effects():
    """把所有可测因素折算成同一尺度的效应量（Dice 点）。"""
    eff = []
    for a in TGTS:
        raw = [BEST[(a, k)][2] for k in LEVELS]
        cal = [BEST[(a, k)][1] for k in LEVELS]
        eff.append((f"Evaluation target $V\\geq$1–4, arm {ARM_SHORT.get(a,a)} (fixed thr. 0.5)",
                    "Protocol", max(raw) - min(raw)))
        eff.append((f"Evaluation target $V\\geq$1–4, arm {ARM_SHORT.get(a,a)} (calibrated)",
                    "Protocol", max(cal) - min(cal)))
        for k in LEVELS:
            t, v, v05 = BEST[(a, k)]
            if v - v05 > 0.005:
                eff.append((f"Threshold, arm {ARM_SHORT.get(a,a)} @ {LVL_SHORT[k]} "
                            f"(0.5 vs {t:.2f})", "Protocol", v - v05))
    for k in LEVELS:
        # 论文口径：5 个监督靶 (A-E)，各自校准阈值（§4.3/Table 4, against G4 = 2.99）
        SUP = [a for a in TGTS if a in ("tgtA", "tgtB", "tgtC", "tgtD", "tgtE")]
        vs = [BEST[(a, k)][1] for a in SUP]
        eff.append((f"Training target (range over {len(SUP)} arms) @ {LVL_SHORT[k]}",
                    "Method", max(vs) - min(vs)))
    archs = [a for a in ARMS if a.startswith("arch")]
    if archs:
        pool = archs + (["tgtB"] if "tgtB" in ARMS else [])
        vs = [BEST[(a, 2)][1] for a in pool]
        eff.append((f"Architecture (range over {len(pool)} models) @ $V\\geq2$",
                    "Model", max(vs) - min(vs)))
    else:
        eff += [("Architecture: plain U-Net vs ResNet-34+ImageNet [prior run]", "Model",
                 0.8789 - 0.8518),
                ("Loss design: 6-term vs 5-term, BRBC removed [prior run]", "Model",
                 0.8834 - 0.8789),
                ("Loss design: range over 4 weight settings [prior run]", "Model",
                 0.8768 - 0.8751),
                ("Architecture: ResNet-18 vs ResNet-34 [prior run]", "Model", 0.0004)]
    return eff


def fig3(eff):
    e = sorted(eff, key=lambda r: r[2])
    col = {"Protocol": C_BLUE, "Method": C_ORANGE, "Model": C_AQUA}
    fig, ax = plt.subplots(figsize=(COL2, 0.185 * len(e) + 0.95))
    y = np.arange(len(e))
    ax.barh(y, [x[2] * 100 for x in e], height=0.66, color=[col[x[1]] for x in e], zorder=3)
    for i, x in enumerate(e):
        ax.text(x[2] * 100 + 0.2, i, f"{x[2]*100:.2f}", va="center", fontsize=6.4, color=INK)
    ax.set_yticks(y, [x[0] for x in e], fontsize=6.6)
    xlab = "Effect size (Dice points)"
    ax.set_xlabel(xlab, fontsize=7)
    # 长说明放轴外底部（右对齐、可换行），避免超出右边缘被截断
    if NOISE_SD:
        _note = (f"Shaded band: run-to-run noise, $2\\sigma$ = {2*NOISE_SD*100:.2f} pts "
                 f"({NOISE_ARMS} arms $\\times$ {NOISE_SEEDS} seeds).  "
                 f"Dashed line: largest non-protocol effect.")
        ax.text(1.0, -0.075, _note, transform=ax.transAxes, ha="right", va="top",
                fontsize=6.2, color=INK2, linespacing=1.4)
    ax.set_xlim(0, max(x[2] for x in e) * 100 * 1.13)
    style(ax, "x")
    if NOISE_SD:
        # 2x 单次训练 SD 以下的差异与随机重跑不可区分，画成灰带
        ax.axvspan(0, 2 * NOISE_SD * 100, color="#f0efea", zorder=1)

    p_max = max(x[2] for x in e if x[1] == "Protocol")
    m_max = max(x[2] for x in e if x[1] in ("Model", "Method"))
    ax.axvline(m_max * 100, color=INK3, lw=0.8, ls=(0, (3, 2)), zorder=2)

    ax.legend(handles=[Patch(fc=col[c], label=c) for c in ("Protocol", "Method", "Model")],
              loc="lower right", ncol=3, bbox_to_anchor=(1.0, -0.02))
    ax.set_title(f"Protocol choices dominate model choices  "
                 f"({p_max*100:.1f} vs {m_max*100:.1f} Dice points, {p_max/m_max:.1f}$\\times$)",
                 loc="left", pad=6)
    save(fig, "fig3_effect_sizes")
    return p_max, m_max


# ================================================================ Fig. 4
def fig4():
    """训练靶 x 评测靶交叉矩阵。"""
    rows = TGTS
    M = np.array([[BEST[(a, k)][1] for k in LEVELS] for a in rows])
    T = np.array([[BEST[(a, k)][0] for k in LEVELS] for a in rows])
    fig, ax = plt.subplots(figsize=(COL1 + 1.15, 0.44 * len(rows) + 1.5))
    im = ax.imshow(M, cmap="Blues", vmin=M.min() - 0.015, vmax=M.max() + 0.005)
    for i in range(len(rows)):
        for j in range(4):
            ax.text(j, i - 0.10, f"{M[i,j]:.4f}", ha="center", va="center", fontsize=7.2,
                    color="white" if M[i, j] > M.mean() else INK, zorder=3)
            tt = f"{T[i,j]:.3f}" if (T[i, j] < 0.01 or T[i, j] > 0.99) else f"{T[i,j]:.2f}"
            ax.text(j, i + 0.24, f"thr {tt}", ha="center", va="center", fontsize=5.8,
                    color="#dbe7f5" if M[i, j] > M.mean() else INK2, zorder=3)
    ax.set_xticks(range(4), [LVL_SHORT[k] for k in LEVELS])
    ax.set_yticks(range(len(rows)), [lab(a) for a in rows], fontsize=7)
    ax.set_xlabel("Evaluation target")
    ax.set_ylabel("Training target")
    ax.set_title("Dice at each arm's own calibrated threshold", loc="left", pad=6)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.ax.tick_params(labelsize=6, width=0.5, length=2)
    cb.outline.set_linewidth(0.5)
    save(fig, "fig4_cross_matrix")
    return M, T


# ================================================================ Fig. 5
def fig5():
    """森林图：三个关键对比各占一个分栏，每栏四个评测靶。

    v2 改动：原来把 12 行塞进一个坐标区、分组名沿 y 轴竖排，结果三个组名
    互相重叠且右侧留白过大。改成三个共享 x 轴的纵向分栏，组名放在各栏标题。
    """
    cons = [c for c in CONTRASTS if c[0] in ARMS and c[1] in ARMS]
    if not cons and len(ARMS) >= 2:                      # 兼容只有两个臂的旧数据
        cons = [(ARMS[0], ARMS[1],
                 f"{ARM_SHORT.get(ARMS[0], ARMS[0])} $-$ {ARM_SHORT.get(ARMS[1], ARMS[1])}")]
    if not cons:
        print("  -> fig5 跳过（可对比的臂不足）")
        return []

    data, out = [], []
    for a_, b_, name in cons:
        rows = []
        for k in LEVELS:
            ta, tb = BEST[(a_, k)][0], BEST[(b_, k)][0]
            per = []
            for f_ in FOLDS:
                s_ = df[df.fold == f_]
                per.append(sub(s_, a_, k, ta)[f"dice_v{k}"].mean()
                           - sub(s_, b_, k, tb)[f"dice_v{k}"].mean())
            ga = sub(df, a_, k, ta).groupby(["fold", "case_id"])[f"dice_v{k}"].mean()
            gb = sub(df, b_, k, tb).groupby(["fold", "case_id"])[f"dice_v{k}"].mean()
            ga, gb = ga.align(gb, join="inner")
            d = (ga - gb).values
            rng = np.random.default_rng(0)
            lo, hi = np.percentile([rng.choice(d, len(d), replace=True).mean()
                                    for _ in range(5000)], [2.5, 97.5])
            r = dict(contrast=name, a=a_, b=b_, k=k, per=per, m=float(d.mean()),
                     lo=float(lo), hi=float(hi), n=int(len(d)),
                     p=float(stats.ttest_rel(ga, gb).pvalue))
            rows.append(r); out.append(r)
        data.append((name, rows))

    # x 轴范围由所有点决定，右侧只留放数值标签的固定宽度
    allv = [v * 100 for _, rows in data for r in rows for v in r["per"]] + \
           [r["lo"] * 100 for _, rows in data for r in rows] + \
           [r["hi"] * 100 for _, rows in data for r in rows]
    x0, x1 = min(allv), max(allv)
    pad = (x1 - x0) * 0.06
    span = (x1 + pad) - (x0 - pad)
    xlim = (x0 - pad, x1 + pad + span * 0.62)            # 右侧 62% 留给数值标签

    fig, axes = plt.subplots(len(data), 1, figsize=(COL2 * 0.78, 1.28 * len(data) + 0.75),
                             sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (name, rows) in zip(axes, data):
        ax.axvline(0, color=INK2, lw=0.9, zorder=2)
        for i, r in enumerate(rows):
            y = len(rows) - 1 - i
            ax.plot([v * 100 for v in r["per"]], [y + 0.17] * len(r["per"]), "o", ms=3.0,
                    color=INK3, alpha=0.85, mec="white", mew=0.4, zorder=4)
            ax.plot([r["lo"] * 100, r["hi"] * 100], [y - 0.14] * 2, "-",
                    color=C_BLUE, lw=1.6, zorder=4)
            ax.plot([r["m"] * 100], [y - 0.14], "D", ms=4.4, color=C_BLUE,
                    mec="white", mew=0.7, zorder=5)
            star = "" if r["p"] >= 0.05 else ("***" if r["p"] < 1e-3 else "*")
            ax.text(x1 + pad + span * 0.04, y,
                    f"{r['m']*100:+.2f}  [{r['lo']*100:+.2f}, {r['hi']*100:+.2f}] {star}",
                    ha="left", va="center", fontsize=6.3, color=INK)
        ax.set_yticks(range(len(rows)), [LVL_SHORT[r["k"]] for r in rows][::-1], fontsize=7)
        ax.set_ylim(-0.62, len(rows) - 0.38)
        ax.set_xlim(*xlim)
        ax.set_title(name, loc="left", pad=3, fontsize=7.6)
        style(ax, "x")
    axes[-1].set_xlabel("Dice difference (points).  Positive favours the first arm.")
    fig.legend(handles=[plt.Line2D([], [], ls="none", marker="o", ms=3.0, color=INK3,
                                   label="individual fold"),
                        plt.Line2D([], [], marker="D", ms=4.4, color=C_BLUE,
                                   label="pooled, 95% CI (case-level bootstrap)")],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.16 / len(data)))
    save(fig, "fig5_forest")
    return out


# ================================================================ Fig. 6
def fig6():
    """校准失配的直接证据。"""
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.35))
    ax = axes[0]
    for a in TGTS:
        g = (df[df.arm == a].groupby(["threshold", "case_id"]).area_pred.mean()
             .groupby("threshold").mean())
        ax.plot(g.index, g.values, color=ARM_COLOR.get(a, INK2), lw=1.5, zorder=4, label=lab(a))
    for k, ls, dy in zip((1, 2), ((0, (4, 2)), (0, (1.5, 1.5))), (1.10, 0.72)):
        v = free[f"gt_area_v{k}"].mean()
        ax.axhline(v, color=INK3, lw=0.8, ls=ls, zorder=2)
        ax.text(0.02, v * dy, f"GT {LVL_SHORT[k]} = {v:.0f} px", ha="left", va="center",
                fontsize=6.4, color=INK2, bbox=dict(fc="white", ec="none", pad=1.2))
    ax.axvline(0.5, color=INK3, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.set_xlabel("Binarisation threshold")
    ax.set_ylabel("Mean predicted area (px)")
    ax.set_xlim(0, 1)
    # 四条曲线本身靠得很近，对数轴会把有信息的区间压平；改线性并裁掉
    # 极端阈值处的塌陷段（那里 Dice 已崩，不是关注区间）。
    hi = max(df[df.arm == a].groupby(["threshold", "case_id"]).area_pred.mean()
             .groupby("threshold").mean().max() for a in TGTS)
    ax.set_ylim(60, hi * 1.10)
    ax.set_title("(a)  Predicted area vs threshold", loc="left", pad=4)
    style(ax, "both")
    # 图例移出曲线区域：底部集中图例（9 项 3 行），避免遮挡左侧曲线
    ax_handles, ax_labels = axes[0].get_legend_handles_labels()

    ax = axes[1]
    # 8 根柱并排，用完整臂名会挤成一团；只保留字母代号，图例含义在正文与 Fig.4 中已给出
    labels = [f"GT\n{LVL_SHORT[k]}" for k in LEVELS] + [
        ARM_SHORT.get(a, a).split()[0] for a in TGTS]
    vals = [free[f"gt_area_v{k}"].mean() for k in LEVELS] + [
        free[free.arm == a].prob_mass.mean() for a in TGTS]
    cols = [INK3] * 4 + [ARM_COLOR.get(a, INK2) for a in TGTS]
    b = ax.bar(range(len(vals)), vals, width=0.62, color=cols, zorder=3)
    for bb, v in zip(b, vals):
        ax.text(bb.get_x() + bb.get_width() / 2, v + max(vals) * 0.015, f"{v:.0f}",
                ha="center", va="bottom", fontsize=6.5, color=INK)
    ax.set_xticks(range(len(vals)), labels, fontsize=6.8)
    ax.set_xlabel("Ground truth (left) vs trained arm (right)", labelpad=2)
    ax.set_ylabel("Area or probability mass (px)")
    ax.set_ylim(0, max(vals) * 1.18)
    ax.set_title("(b)  Total probability mass vs ground-truth area", loc="left", pad=4)
    style(ax)
    fig.legend(ax_handles, ax_labels, loc="lower center", ncol=4, fontsize=6.2,
               bbox_to_anchor=(0.5, -0.02), frameon=False, labelspacing=0.5,
               columnspacing=1.4, handlelength=1.2)
    fig.subplots_adjust(bottom=0.34)
    save(fig, "fig6_calibration")


print("\n生成插图：")
fig1(); fig2()
eff = effects()
p_max, m_max = fig3(eff)
M, T = fig4()
forest = fig5()
fig6()

summary = {
    "n_folds": len(FOLDS), "n_cases": N_CASES,
    "n_samples": int(len(sub(df, ARMS[0], 1, 0.5))),
    "arms": ARMS, "training_arms": TGTS,
    "protocol_max_pts": p_max * 100, "model_max_pts": m_max * 100, "ratio": p_max / m_max,
    "seed_noise_sd_pts": (NOISE_SD * 100) if NOISE_SD else None,
    "seed_noise_arms": NOISE_ARMS, "seed_noise_seeds": NOISE_SEEDS,
    "matrix": {a: {int(k): {"thr": BEST[(a, k)][0], "dice_opt": BEST[(a, k)][1],
                            "dice_05": BEST[(a, k)][2]} for k in LEVELS} for a in ARMS},
    "n_valid": {int(k): int(len(sub(df, ARMS[0], k, 0.5))) for k in LEVELS},
    "forest": forest,
    "effects": [{"factor": n, "type": t, "pts": v * 100} for n, t, v in
                sorted(eff, key=lambda r: -r[2])],
    "gt_area": {f"v{k}": float(free[f"gt_area_v{k}"].mean()) for k in LEVELS},
    "prob_mass": {a: float(free[free.arm == a].prob_mass.mean()) for a in ARMS},
    "soft_dice": {a: float(free[free.arm == a].soft_dice.mean()) for a in ARMS},
    "brier": {a: float(free[free.arm == a].brier.mean()) for a in ARMS},
    "per_fold_v2": {a: sub(df, a, 2, BEST[(a, 2)][0]).groupby(["fold", "case_id"]).dice_v2.mean()
                    .groupby("fold").mean().to_dict()
                    for a in ARMS},
}
with open(os.path.join(OUT, "figure_values.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"\n关键数值 -> {OUT}/figure_values.json")
