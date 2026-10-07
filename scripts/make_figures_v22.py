# -*- coding: utf-8 -*-
"""make_figures_v22.py —— 论文 v22 的全部 9 张插图，一个脚本、全部从数据现取（2026-10-02）。

为什么重写：v21 稿里嵌入的 9 张图来自 5 个不同脚本，其中 Fig. 1/2/4/5/6/8 的生成代码
已不在仓库里（make_paper_figures.py 产出的是另一套设计和文件名），且有三张图的数字过期：
Fig. 5 的噪声底仍是作废的实例级 1.15、Fig. 3 的排序与误差线和图注不符、Fig. 7/8 用的是
未按病例级重新校准的阈值。本脚本把 9 张图全部固化到一处，任何数字都从权威 JSON/CSV 读取，
图上出现的关键数值同时写入 figure_values_v22.json 供正文核对。

输入（均为仓库内的真实实验产物）：
  outputs/analysis_full/symmetric_analysis.json         交叉验证：Table 3/4、面积匹配
  outputs/analysis_full/noise_floor.json                噪声底 2σ 与 CI
  outputs/analysis_test/symmetric_analysis_applied.json 保留测试集（阈值从验证集导入）
  outputs/analysis/effect_uncertainty.json              Fig. 3 各分量的病例级 bootstrap CI
  outputs/analysis_scope/offset_paired_stats.json       十臂位移中位数（scripts/offset_paired_stats.py）
  outputs/analysis_scope/diag_offset_allarms.json       Fig. 7 位移曲线（fold0，各臂校准阈值）
  outputs/analysis_scope/scope_analysis_merged2.json    Fig. 3/8 视野阶梯（各臂校准阈值）
  outputs/figures/v22/source_data/*                     Fig. 2/6 曲线（scripts/make_figure_source_data.py）
  outputs/analysis/{disagreement_index,qubiq_all_effects,generalisation_table}.json   Fig. 9
  data/processed/patches/LIDC-IDRI-0717.npz             Fig. 1 的 CT patch 与投票图
输出：
  outputs/figures/v22/Fig1.png … Fig9.png（600 dpi，嵌入 Word 用）与同名 .pdf（矢量，终稿用）
  outputs/figures/v22/figure_values_v22.json

配色：dataviz 技能的验证色板（validate_palette.js：四臂 蓝/橙/紫/青 全配对通过；
Fig. 8 原来的 黄/橙 组合正常视觉 ΔE 13.7 < 15 不合格，改为 青）。文字一律用墨色，颜色只给数据标记。
用法（仓库根目录）：
    python scripts/make_figure_source_data.py    # 先聚合 Fig. 2/6 的曲线（需要逐样本大 CSV）
    python scripts/make_figures_v22.py
"""
import itertools
import json
import os
import sys

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator

ROOT = os.environ.get("FIG_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("FIG_OUT") or os.path.join(ROOT, "outputs", "figures", "v22")
os.makedirs(OUT, exist_ok=True)

W = 6.5                                    # 版心宽度（英寸），= Word 稿中的嵌入宽度 16.51 cm
BLUE, ORANGE, VIOLET, AQUA, YELLOW = "#2a78d6", "#eb6834", "#4a3aa7", "#1baf7a", "#eda100"
INK, INK2, INK3, GRID, BAND = "#22221f", "#52514e", "#8a8983", "#e6e5e0", "#efeee9"
GREY_LINE = "#b9b8b1"
ARM_COLOR = {"tgtA": BLUE, "tgtB": ORANGE, "tgtC": VIOLET, "tgtD": AQUA}
ARM_LABEL = {"tgtA": "A  union", "tgtB": "B  majority", "tgtC": "C  majority + soft",
             "tgtD": "D  soft only", "tgtE": "E  SVLS"}
MINUS = "−"

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK3,
    "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK, "text.color": INK,
    "legend.frameon": False, "figure.dpi": 120, "savefig.dpi": 600,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.10,
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": True,
})

VALUES = {}          # 图上出现的数字 -> figure_values_v22.json


def J(*p):
    with open(os.path.join(ROOT, *p), encoding="utf-8") as f:
        return json.load(f)


def sgn(x, nd=2):
    """带真减号的有符号数字。"""
    s = f"{x:+.{nd}f}"
    return s.replace("-", MINUS)


def num(x, nd=2):
    return f"{x:.{nd}f}".replace("-", MINUS)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  -> {name}.png / .pdf")


def grid(ax, axis="y"):
    ax.grid(True, axis=axis, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def panel_tag(ax, s, x=-0.02, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="right", color=INK)


# ====================================================================== Fig. 1
def fig1():
    z = np.load(os.path.join(ROOT, "data", "processed", "patches", "LIDC-IDRI-0717.npz"))
    idx = 7
    img, vote = z["images"][idx].astype(float), z["votes"][idx]
    counts = [int((vote >= k).sum()) for k in (1, 2, 3, 4)]
    assert counts == [92, 53, 41, 36], counts            # 与图注一致
    ys, xs = np.nonzero(vote)
    cy, cx = int(round(ys.mean())), int(round(xs.mean()))
    h = 24                                                # 48 × 48 视窗
    sl = (slice(cy - h, cy + h), slice(cx - h, cx + h))
    im_c, v_c = img[sl], vote[sl]
    eq_d_mm = 2 * np.sqrt(counts[1] / np.pi) * 0.6       # 多数票掩膜等效直径，名义 0.6 mm/px
    VALUES["fig1"] = {"case": "LIDC-IDRI-0717", "patch_index": idx, "px_V>=k": counts,
                      "union_over_unanimous": round(counts[0] / counts[3], 2),
                      "equivalent_diameter_mm_nominal": round(eq_d_mm, 2), "crop_px": 2 * h}

    fig = plt.figure(figsize=(W, 1.62))
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 0.075], hspace=0.10, wspace=0.07,
                          left=0.005, right=0.995, top=0.88, bottom=0.10)
    titles = ["(a)  CT patch", "(b)  vote map $V(x)$", "(c)  $G_1$  union",
              "(d)  $G_2$  majority", "(e)  $G_3$", "(f)  $G_4$  unanimous"]
    cmap_v = mpl.colors.ListedColormap(plt.get_cmap("magma")(np.linspace(0.0, 0.97, 5)))
    for i in range(6):
        ax = fig.add_subplot(gs[0, i])
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color("#c9c8c2"); sp.set_linewidth(0.6)
        ax.set_title(titles[i], fontsize=7.5, pad=3)
        if i == 0:
            ax.imshow(im_c, cmap="gray", vmin=0, vmax=255, interpolation="bilinear")
        elif i == 1:
            imv = ax.imshow(v_c, cmap=cmap_v, vmin=-0.5, vmax=4.5, interpolation="nearest")
        else:
            k = i - 1
            ax.imshow(im_c, cmap="gray", vmin=0, vmax=255, interpolation="bilinear")
            m = np.ma.masked_where(v_c < k, np.ones_like(v_c, dtype=float))
            ax.imshow(m, cmap=mpl.colors.ListedColormap(["#9cc2ef"]), alpha=0.55,
                      interpolation="nearest", vmin=0, vmax=1)
            ax.contour((v_c >= k).astype(float), levels=[0.5], colors=[BLUE], linewidths=1.1)
            cap = fig.add_subplot(gs[1, i]); cap.axis("off")
            cap.text(0.5, 0.5, f"$V \\geq {k}$    {counts[k-1]} px", ha="center", va="center",
                     fontsize=7.3, color=INK)
    cax = fig.add_subplot(gs[1, 1])
    cb = fig.colorbar(imv, cax=cax, orientation="horizontal", ticks=[0, 1, 2, 3, 4])
    cb.ax.tick_params(labelsize=6.5, length=2, width=0.5, pad=1.5, colors=INK2)
    cb.outline.set_linewidth(0.5); cb.outline.set_edgecolor(INK3)
    cb.set_label("number of raters", fontsize=6.8, labelpad=1.5, color=INK2)
    save(fig, "Fig1")


# ====================================================================== Fig. 2
def logit_axis(ax):
    ax.set_xscale("logit")
    ax.xaxis.set_major_locator(FixedLocator([0.01, 0.5, 0.99]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(FixedLocator([0.002, 0.005, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7,
                                              0.8, 0.9, 0.95, 0.98, 0.995, 0.998]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(0.0008, 0.9992)


def fig2():
    c = pd.read_csv(os.path.join(ROOT, "outputs", "figures", "v22", "source_data", "cv_threshold_curves.csv"))
    T = J("outputs", "analysis_full", "symmetric_analysis.json")["table1"]
    arms = ["tgtA", "tgtB", "tgtC", "tgtD"]
    fig, axes = plt.subplots(1, 4, figsize=(W, 2.35), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.90, bottom=0.30, wspace=0.10)
    opt = {}
    for ax, k in zip(axes, (1, 2, 3, 4)):
        for a in arms:
            s = c[c.arm == a].sort_values("threshold")
            ax.plot(s.threshold, s[f"dice_v{k}"], color=ARM_COLOR[a], lw=1.4, zorder=3)
            t = float(T[a][str(k)]["thr"])
            v = float(s.loc[np.isclose(s.threshold, t), f"dice_v{k}"].iloc[0])
            assert abs(v - T[a][str(k)]["dice_opt"]) < 1e-5, (a, k, v)
            assert np.isclose(s.threshold.iloc[s[f"dice_v{k}"].values.argmax()], t), (a, k)
            ax.plot([t], [v], "o", ms=4.6, color=ARM_COLOR[a], mec="white", mew=0.8, zorder=5)
            opt[f"{a}_G{k}"] = {"thr": t, "dice": round(v, 4)}
        ax.axvline(0.5, color=INK3, lw=0.7, ls=(0, (3, 2)), zorder=1)
        logit_axis(ax)
        ax.set_title(f"reference $G_{k}$", pad=4)
        grid(ax)
    axes[0].set_ylabel("Dice")
    axes[0].set_ylim(0.15, 0.93)
    fig.text(0.535, 0.155, "Binarisation threshold (logit scale)", ha="center", fontsize=7.5)
    h = [Line2D([], [], color=ARM_COLOR[a], lw=1.6, label=ARM_LABEL[a]) for a in arms]
    h += [Line2D([], [], ls="none", marker="o", ms=4.6, color=INK3, mec="white", label="optimum"),
          Line2D([], [], color=INK3, lw=0.7, ls=(0, (3, 2)), label="threshold 0.5")]
    fig.legend(handles=h, loc="lower center", ncol=6, bbox_to_anchor=(0.535, -0.01),
               handlelength=1.5, columnspacing=1.2, fontsize=6.8)
    VALUES["fig2"] = opt
    save(fig, "Fig2")


# ====================================================================== Fig. 3
def fig3():
    T = J("outputs", "analysis_test", "symmetric_analysis_applied.json")["table1"]
    E = J("outputs", "analysis", "effect_uncertainty.json")["own_sample_bootstrap"]
    O = J("outputs", "analysis_scope", "offset_paired_stats.json")["ten_arms_fold0"]
    N = J("outputs", "analysis_full", "noise_floor.json")
    S = J("outputs", "analysis_scope", "scope_analysis_merged2.json")["table4_paired"]

    def fov(arm):
        r = [x for x in S if x["arm"] == arm and x["level"] == 2 and x["segment"] == "纯视野效应"]
        assert len(r) == 1
        return -r[0]["diff"], (-r[0]["ci"][1], -r[0]["ci"][0])

    def span(arms, k):
        v = [T[a][str(k)]["dice_opt"] for a in arms]
        return 100 * (max(v) - min(v))

    # 点估计与 CI 的双重来源互相核对（effect_uncertainty 与 symmetric/scope 两套独立代码）
    sup = span(["tgtA", "tgtB", "tgtC", "tgtD", "tgtE"], 4)
    arch = span(["tgtB", "archR18", "archCNX", "archPVT", "archPU1e3"], 2)
    tgt = 100 * (max(T["tgtA"][k]["dice_05"] for k in "1234") - min(T["tgtA"][k]["dice_05"] for k in "1234"))
    thr = 100 * (T["tgtA"]["4"]["dice_opt"] - T["tgtA"]["4"]["dice_05"])
    shift = 100 * (T["tgtB"]["2"]["dice_opt"] - T["tgtBshift"]["2"]["dice_opt"])
    fov_pi, fov_pi_ci = fov("tgtBshift")
    fov_m, fov_m_ci = fov("tgtBshift_bg")
    for name, pt in [("supervision_range_pts", sup), ("architecture_range_pts", arch),
                     ("evaluation_target_span_pts", tgt), ("threshold_effect_pts", thr),
                     ("translation_cost_pts", shift), ("fov_position_invariant_pts", -fov_pi)]:
        assert abs(abs(E[name]["point"]) - abs(pt)) < 0.006, (name, E[name]["point"], pt)
    assert abs(abs(E["fov_effect_pts"]["point"]) - fov_m) < 0.006
    drops = sorted(v["drop8_pts"] for v in O["arms"].values())
    noise = N["fixed_0_5"]["two_sigma_pts"]
    rows = [
        ("Evaluation field of view\n128 → 512 px, translation-augmented arm", fov_pi, fov_pi_ci, "protocol", None),
        ("Evaluation target\n$V\\geq1$ to $V\\geq4$, arm A at threshold 0.5", tgt,
         tuple(E["evaluation_target_span_pts"]["ci95"]), "protocol", None),
        ("Lesion offset of 8 px from the patch centre\nmedian of ten arms (dots: each arm)",
         O["median_drop8_pts"], None, "data", drops),
        ("Evaluation field of view\n128 → 512 px, both training confounds removed", fov_m, fov_m_ci, "protocol", None),
        ("Binarisation threshold\narm A against $V\\geq4$", thr, tuple(E["threshold_effect_pts"]["ci95"]), "protocol", None),
        ("Cost of translation augmentation\non the centred protocol", shift,
         tuple(E["translation_cost_pts"]["ci95"]), "data", None),
        ("Supervision target\nfive arms, against $V\\geq4$", sup, tuple(E["supervision_range_pts"]["ci95"]), "method", None),
        ("Architecture\nfive models, 8× parameter range", arch, tuple(E["architecture_range_pts"]["ci95"]), "method", None),
    ]
    rows.sort(key=lambda r: -r[1])
    rows.append(("Random seed (2σ)\npooled over four references", noise, tuple(N["fixed05_two_sigma_ci"]), "noise", None))
    role_c = {"protocol": BLUE, "data": VIOLET, "method": ORANGE, "noise": "#8f8f8a"}

    fig, ax = plt.subplots(figsize=(W, 3.45))
    fig.subplots_adjust(left=0.415, right=0.985, top=0.99, bottom=0.12)
    y = np.arange(len(rows))[::-1]
    ax.axvspan(0, noise, color=BAND, zorder=0, lw=0)
    for yi, (lab, v, ci, role, dots) in zip(y, rows):
        ax.barh(yi, v, height=0.58, color=role_c[role], edgecolor="white", lw=0.8, zorder=3)
        right = v
        if ci:
            ax.plot(ci, [yi, yi], color=INK, lw=0.8, zorder=5)
            for b in ci:
                ax.plot([b, b], [yi - 0.12, yi + 0.12], color=INK, lw=0.8, zorder=5)
            right = max(ci[1], v)
        if dots:
            ax.plot(dots, [yi] * len(dots), "o", ms=3.2, color="white", mec=INK, mew=0.6, zorder=6)
            right = max(dots)
        ax.text(right + 0.9, yi, num(v), va="center", ha="left", fontsize=7.3, fontweight="bold", zorder=6)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.9, linespacing=1.3)
    ax.tick_params(axis="y", length=0, colors=INK)
    ax.set_xlim(0, 51)
    ax.set_ylim(y[-1] - 0.6, y[0] + 0.6)
    ax.set_xlabel("Change in Dice (points) when one factor is varied, all others held fixed")
    grid(ax, "x")
    ax.spines["left"].set_visible(False)
    ax.legend(handles=[Patch(facecolor=role_c[k], label=t) for k, t in
                       [("protocol", "Evaluation protocol"), ("data", "Data/training protocol"),
                        ("method", "Learning design"), ("noise", "Run-to-run noise (2σ)")]],
              loc="lower right", bbox_to_anchor=(1.0, 0.02), handlelength=1.0, handleheight=0.9,
              labelspacing=0.35, fontsize=6.8)
    VALUES["fig3"] = [{"row": r[0].replace("\n", " | "), "value": round(r[1], 2),
                       "ci95": None if r[2] is None else [round(c, 2) for c in r[2]],
                       "dots": None if r[4] is None else [round(d, 2) for d in r[4]]} for r in rows]
    save(fig, "Fig3")


# ====================================================================== Fig. 4
def fig4():
    T = J("outputs", "analysis_full", "symmetric_analysis.json")["table1"]
    rows = ["tgtA", "tgtB", "tgtC", "tgtD", "tgtE"]
    M = np.array([[100 * T[a][str(k)]["dice_opt"] for k in (1, 2, 3, 4)] for a in rows])
    lo, hi = np.floor(M.min()) - 1.0, np.ceil(M.max())
    fig, ax = plt.subplots(figsize=(W * 0.80, 2.35))
    im = ax.imshow(M, cmap="Blues", vmin=lo - 6, vmax=hi, aspect="auto")   # 浅端留白，避免最低值发白
    best = M.argmax(axis=0)
    for i in range(len(rows)):
        for j in range(4):
            dark = (M[i, j] - (lo - 6)) / (hi - (lo - 6)) > 0.55
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7.4,
                    color="white" if dark else INK,
                    fontweight="bold" if best[j] == i else "normal")
            if best[j] == i:
                ax.add_patch(mpl.patches.Rectangle((j - 0.48, i - 0.46), 0.96, 0.92, fill=False,
                                                   ec=INK, lw=0.9, zorder=4))
    ax.set_xticks(range(4), [f"$G_{k}$" for k in (1, 2, 3, 4)])
    ax.set_yticks(range(len(rows)), [ARM_LABEL[a] for a in rows])
    ax.tick_params(length=0, colors=INK)
    ax.set_xlabel("Evaluation reference (each cell at that arm's own calibrated threshold)")
    ax.set_ylabel("Supervision target")
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.ax.set_ylim(lo, hi)
    cb.set_ticks(np.arange(np.ceil(lo / 4) * 4, hi + 0.01, 4))
    cb.ax.tick_params(labelsize=6.5, length=2, width=0.5, colors=INK2)
    cb.outline.set_visible(False)
    cb.set_label("Dice (points)", fontsize=7, color=INK2)
    col_range = M.max(axis=0) - M.min(axis=0)
    VALUES["fig4"] = {"matrix": {a: [round(v, 2) for v in M[i]] for i, a in enumerate(rows)},
                      "best_per_column": [rows[b] for b in best],
                      "within_column_range": [round(v, 2) for v in col_range]}
    save(fig, "Fig4")


# ====================================================================== Fig. 5
def fig5():
    S = J("outputs", "analysis_full", "symmetric_analysis.json")["table2"]
    noise = J("outputs", "analysis_full", "noise_floor.json")["fixed_0_5"]["two_sigma_pts"]
    groups = [("tgtC-tgtA", "C − A   consensus vs union  (mask extent and soft supervision)", BLUE),
              ("tgtB-tgtA", "B − A   mask extent only", ORANGE),
              ("tgtC-tgtB", "C − B   soft supervision only", VIOLET)]
    fig, ax = plt.subplots(figsize=(W, 3.25))
    y, yt, yl, out = 0.0, [], [], {}
    title_y = []
    for key, title, col in groups:
        title_y.append((y, title, col))
        y -= 0.85
        for k in (4, 3, 2, 1):
            r = S[key][str(k)]
            d, lo_, hi_ = 100 * r["diff"], 100 * r["ci"][0], 100 * r["ci"][1]
            null = lo_ <= 0 <= hi_
            ax.plot([lo_, hi_], [y, y], color=col, lw=1.8, solid_capstyle="round", zorder=4)
            ax.plot([d], [y], "o", ms=4.8, color=col, mec="white", mew=0.9, zorder=5)
            ax.text(2.45, y, f"{sgn(d)}  [{sgn(lo_)}, {sgn(hi_)}]   n = {r['n']}", va="center",
                    ha="left", fontsize=6.9, color=INK3 if null else INK)
            yt.append(y); yl.append(f"$G_{k}$")
            out[f"{key}_G{k}"] = {"diff": round(d, 2), "ci": [round(lo_, 2), round(hi_, 2)], "n": r["n"],
                                  "ci_includes_zero": bool(null)}
            y -= 0.62
        y -= 0.35
    for ty, title, col in title_y:
        ax.plot([-3.85], [ty - 0.02], "s", ms=5.0, color=col, clip_on=False, zorder=6)
        ax.text(-3.72, ty, title, va="center", ha="left", fontsize=7.3, fontweight="bold", color=INK)
    ax.axvspan(-noise, noise, color=BAND, zorder=0, lw=0)
    ax.axvline(0, color=INK2, lw=0.8, zorder=2)
    ax.set_yticks(yt, yl, fontsize=7.2)
    ax.tick_params(axis="y", length=0, colors=INK)
    ax.set_xlim(-3.9, 5.75)
    ax.set_xticks([-3, -2, -1, 0, 1, 2])
    ax.set_ylim(y + 0.2, 0.35)
    ax.spines["left"].set_visible(False)
    grid(ax, "x")
    ax.set_xlabel("Dice difference (points); positive favours the first arm\n"
                  f"shaded band: run-to-run noise floor, ±{noise:.2f} points (2σ)", linespacing=1.4)
    ax.xaxis.set_label_coords(0.37, -0.085)
    VALUES["fig5"] = {"noise_two_sigma": noise, "contrasts": out}
    save(fig, "Fig5")


# ====================================================================== Fig. 6
def fig6():
    c = pd.read_csv(os.path.join(ROOT, "outputs", "figures", "v22", "source_data", "cv_threshold_curves.csv"))
    R = J("outputs", "figures", "v22", "source_data", "cv_reference_areas.json")
    T5 = J("outputs", "analysis_full", "symmetric_analysis.json")["table5"]
    arms = ["tgtA", "tgtB", "tgtC", "tgtD"]
    g1, g2 = R["gt_area_case_mean_px"]["G1"], R["gt_area_case_mean_px"]["G2"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(W, 2.45), gridspec_kw={"width_ratios": [1.45, 1]})
    fig.subplots_adjust(left=0.085, right=0.99, top=0.90, bottom=0.30, wspace=0.30)
    for a in arms:
        s = c[c.arm == a].sort_values("threshold")
        ax.plot(s.threshold, s.area_case_mean, color=ARM_COLOR[a], lw=1.4, zorder=3)
        t = float(T5[a]["threshold"])
        v = float(s.loc[np.isclose(s.threshold, t), "area_case_mean"].iloc[0])
        assert abs(v - T5[a]["area"]) < 0.05, (a, v, T5[a]["area"])
        ax.plot([t], [v], "o", ms=4.4, color=ARM_COLOR[a], mec="white", mew=0.8, zorder=5)
    for val, ls in ((g1, (0, (4, 2))), (g2, (0, (1.2, 1.4)))):
        ax.axhline(val, color=INK2, lw=0.8, ls=ls, zorder=2)
    ax.text(0.9985, g1 + 5, f"$G_1$ mean area {g1:.1f} px", ha="right", va="bottom", fontsize=6.8, color=INK2)
    ax.text(0.00095, g2 - 6, f"$G_2$ mean area {g2:.1f} px", ha="left", va="top", fontsize=6.8, color=INK2)
    logit_axis(ax)
    ax.set_ylim(0, 310)
    ax.set_yticks([0, 50, 100, 150, 200, 250, 300])
    ax.set_xlabel("Binarisation threshold (logit scale)")
    ax.set_ylabel("Mean predicted area (px)")
    grid(ax)
    panel_tag(ax, "a", x=-0.12)

    brier = [1000 * R["brier_instance_mean"][a] for a in arms]
    b = bx.bar(range(4), brier, width=0.58, color=[ARM_COLOR[a] for a in arms], zorder=3)
    for bb, v in zip(b, brier):
        bx.text(bb.get_x() + bb.get_width() / 2, v + 0.06, f"{v:.2f}", ha="center", va="bottom",
                fontsize=7.2, fontweight="bold")
    bx.set_xticks(range(4), ["A", "B", "C", "D"])
    bx.set_xlabel("Supervision target")
    bx.set_ylabel("Brier score vs $p = V/4$  (×10$^{-3}$)")
    bx.set_ylim(0, 3.7)
    grid(bx)
    panel_tag(bx, "b", x=-0.2)
    h = [Line2D([], [], color=ARM_COLOR[a], lw=1.6, label=ARM_LABEL[a]) for a in arms]
    h += [Line2D([], [], ls="none", marker="o", ms=4.4, color=INK3, mec="white", label="area-matched point")]
    fig.legend(handles=h, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.01), fontsize=6.8,
               handlelength=1.5, columnspacing=1.2)
    VALUES["fig6"] = {"gt_case_mean_px": {"G1": round(g1, 1), "G2": round(g2, 1)},
                      "area_matched": {a: {"thr": T5[a]["threshold"], "area": round(T5[a]["area"], 1)} for a in arms},
                      "brier_x1000": {a: round(v, 2) for a, v in zip(arms, brier)}}
    save(fig, "Fig6")


# ====================================================================== Fig. 7
def fig7():
    d = J("outputs", "analysis_scope", "diag_offset_allarms.json")
    P = J("outputs", "analysis_scope", "offset_paired_stats.json")
    offs = [int(o) for o in d["offsets"]]
    runs = d["runs"]
    main_, shift_ = "tgtB_fold0", "tgtBshift_fold0"
    others = [r for r in runs if r not in (main_, shift_)]
    assert len(others) == 9 and not any("shift" in r for r in others), others
    fig, ax = plt.subplots(figsize=(W, 2.75))
    ax.axvspan(0, 32, color=BAND, zorder=0, lw=0)
    for r in others:
        ax.plot(offs, [runs[r]["mean_by_offset"][str(o)] for o in offs], color=GREY_LINE, lw=0.9, zorder=2)
    yb = [runs[main_]["mean_by_offset"][str(o)] for o in offs]
    ys = [runs[shift_]["mean_by_offset"][str(o)] for o in offs]
    ax.plot(offs, yb, color=VIOLET, lw=2.0, marker="o", ms=4.2, mec="white", mew=0.7, zorder=4)
    ax.plot(offs, ys, color=ORANGE, lw=2.0, marker="s", ms=4.0, mec="white", mew=0.7, zorder=4)
    ax.set_xticks(offs)
    ax.set_xlim(-1.5, 57.5)
    ax.set_ylim(0, 0.95)
    ax.set_xlabel("Offset of the evaluation window from the lesion centroid (px)")
    ax.set_ylabel("Dice against $G_2$")
    grid(ax)
    ax.text(16, 0.925, "translation-augmentation range (±32 px)", ha="center", va="top", fontsize=6.9, color=INK2)
    rad = P["eligible_nodules"]["median_equivalent_radius_px"]
    ax.annotate(f"8 px ≈ {8 / rad:.1f} lesion radii", xy=(8, yb[1]), xytext=(15, yb[1] + 0.035),
                fontsize=6.9, color=INK2, va="center", bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2),
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6, shrinkA=0, shrinkB=3))
    h = [Line2D([], [], color=VIOLET, lw=2.0, marker="o", ms=4.2, mec="white", label="arm B (centre-cropped)"),
         Line2D([], [], color=ORANGE, lw=2.0, marker="s", ms=4.0, mec="white",
                label="arm B + translation augmentation"),
         Line2D([], [], color=GREY_LINE, lw=0.9, label="nine further arms without translation augmentation")]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, fontsize=6.8,
              handlelength=1.8, columnspacing=1.2)
    VALUES["fig7"] = {"tgtB_fold0": [round(v, 4) for v in yb], "tgtBshift_fold0": [round(v, 4) for v in ys],
                      "thresholds": {r: runs[r]["thr"] for r in runs}, "n_nodules": runs[main_]["n"]}
    save(fig, "Fig7")


# ====================================================================== Fig. 8
def fig8():
    S = J("outputs", "analysis_scope", "scope_analysis_merged2.json")
    t1, t4 = S["table1_ladder"], S["table4_paired"]
    scopes = ["patch128", "win128", "win192", "win256", "win384", "win512", "slice", "volume"]
    xl = ["patch\n128", "win\n128", "win\n192", "win\n256", "win\n384", "win\n512", "full\nslice", "whole\nvolume"]
    arms = [("tgtB", "B majority (centre-cropped)", VIOLET, "o"),
            ("bg_maj", "+ background sampling", AQUA, "^"),
            ("tgtBshift", "+ translation augmentation", ORANGE, "s"),
            ("tgtBshift_bg", "+ both remedies", BLUE, "D")]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(W, 2.9), gridspec_kw={"width_ratios": [1.55, 1]})
    fig.subplots_adjust(left=0.075, right=0.99, top=0.92, bottom=0.30, wspace=0.42)
    x = np.arange(len(scopes))
    ax.axvspan(0.5, 5.5, color=BAND, zorder=0, lw=0)
    ax.text(3.0, 0.955, "field-of-view ladder", ha="center", va="top", fontsize=6.9, color=INK2)
    vals = {}
    for a, lab, col, mk in arms:
        v = [t1[f"{a}_v2"][s] for s in scopes]
        vals[a] = v
        ax.plot(x, v, color=col, lw=1.5, marker=mk, ms=3.8, mec="white", mew=0.6, zorder=4)
    ax.set_xticks(x, xl, fontsize=6.6)
    ax.set_ylim(0, 0.97)
    ax.set_xlim(-0.4, len(scopes) - 0.6)
    ax.set_ylabel("Dice against $G_2$")
    ax.set_xlabel("Evaluation scope")
    grid(ax)
    panel_tag(ax, "a", x=-0.09)

    y = np.arange(len(arms))[::-1]
    lad = {}
    for yi, (a, lab, col, mk) in zip(y, arms):
        r = [q for q in t4 if q["arm"] == a and q["level"] == 2 and q["segment"] == "纯视野效应"]
        assert len(r) == 1
        loss, ci = -r[0]["diff"], (-r[0]["ci"][1], -r[0]["ci"][0])
        lad[a] = {"loss_pts": round(loss, 2), "ci95": [round(ci[0], 2), round(ci[1], 2)], "p": r[0]["p"]}
        bx.barh(yi, loss, height=0.5, color=col, zorder=3)
        bx.plot(ci, [yi, yi], color=INK, lw=0.8, zorder=5)
        for bnd in ci:
            bx.plot([bnd, bnd], [yi - 0.1, yi + 0.1], color=INK, lw=0.8, zorder=5)
        bx.text(max(ci[1], loss, 0) + 1.5, yi, num(loss), va="center", fontsize=7.3, fontweight="bold")
    bx.axvline(0, color=INK2, lw=0.7)
    bx.set_yticks(y, ["B majority", "+ background", "+ translation", "+ both"])
    bx.tick_params(axis="y", length=0, colors=INK)
    bx.set_xlim(-2, 60)
    bx.set_xlabel("Dice lost, win 128 → win 512 (points)")
    grid(bx, "x")
    bx.spines["left"].set_visible(False)
    panel_tag(bx, "b", x=-0.36)
    h = [Line2D([], [], color=col, lw=1.5, marker=mk, ms=3.8, mec="white", label=lab) for _, lab, col, mk in arms]
    fig.legend(handles=h, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01), fontsize=6.8,
               handlelength=1.8, columnspacing=1.1)
    VALUES["fig8"] = {"ladder_v2": {a: {s: round(t1[f"{a}_v2"][s], 4) for s in scopes} for a, *_ in arms},
                      "ladder_loss": lad}
    save(fig, "Fig8")


# ====================================================================== Fig. 9
def _rank(a):
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a)); s = a[order]; i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and s[j + 1] == s[i]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return r


def _spearman(x, y):
    rx, ry = _rank(x) - _rank(x).mean(), _rank(y) - _rank(y).mean()
    return float(rx @ ry / np.sqrt((rx @ rx) * (ry @ ry)))


def _exact_p(x, y):
    obs = _spearman(x, y)
    rx = _rank(x) - _rank(x).mean(); ry0 = _rank(y)
    hit = tot = 0
    for perm in itertools.permutations(range(len(x))):
        ry = ry0[list(perm)] - ry0.mean()
        r = rx @ ry / np.sqrt((rx @ rx) * (ry @ ry))
        tot += 1; hit += abs(r) >= abs(obs) - 1e-12
    return obs, hit / tot


def fig9():
    dis = J("outputs", "analysis", "disagreement_index.json")
    eff = J("outputs", "analysis", "qubiq_all_effects.json")
    gen = J("outputs", "analysis", "generalisation_table.json")
    # (键, 名称, 副题, 颜色, 标注位置 (x, y), 水平对齐)
    P = [("LIDC", "LIDC-IDRI", "4 raters, CT", BLUE, (0.25, 23.2), "center"),
         ("btum-t2", "brain-tumour t2", "3 raters, MRI", VIOLET, (0.245, 30.3), "left"),
         ("brain-growth", "brain-growth", "7 raters, MRI", ORANGE, (0.49, 20.2), "center"),
         ("prostate-t2", "prostate t2", "6 raters, MRI", AQUA, (0.77, 20.2), "center"),
         ("btum-t3", "brain-tumour t3", "3 raters, MRI", VIOLET, (0.44, 6.4), "center"),
         ("btum-t1", "brain-tumour t1", "3 raters, MRI", VIOLET, (0.715, 7.6), "center"),
         ("kidney", "kidney", "3 raters, CT", ORANGE, (0.875, 11.6), "center"),
         ("prostate-t1", "prostate t1", "6 raters, MRI", AQUA, (1.095, 7.0), "right")]
    dmap = {"brain-growth": "QUBIQ-brain", "kidney": "QUBIQ-kidney", "btum-t1": "QUBIQ-brain-tumor-task1",
            "btum-t2": "QUBIQ-brain-tumor-task2", "btum-t3": "QUBIQ-brain-tumor-task3",
            "prostate-t1": "QUBIQ-prostate-task1", "prostate-t2": "QUBIQ-prostate-task2"}
    emap = dict(dmap, **{"brain-growth": "QUBIQ-brain-growth"})
    pts = {"LIDC": (gen["agreement_index"]["LIDC"], gen["LIDC"]["opt_thr"]["protocol"], gen["LIDC"]["opt_thr"]["supervision"])}
    for k, key in emap.items():
        o = eff[key]["effects"]["opt_thr"]
        pts[k] = (dis[dmap[k]]["area_mean_ratio"], o["protocol"], o["supervision"])
    keys = [p[0] for p in P]
    xs = np.array([pts[k][0] for k in keys]); pr = np.array([pts[k][1] for k in keys]); su = np.array([pts[k][2] for k in keys])
    rho, pval = _exact_p(xs, pr)
    assert abs(rho - (-0.8571)) < 5e-4 and abs(pval - 0.0107) < 5e-4, (rho, pval)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(W, 3.15), gridspec_kw={"width_ratios": [1.12, 0.88]})
    fig.subplots_adjust(left=0.075, right=0.99, top=0.86, bottom=0.14, wspace=0.60)
    for i, (k, name, sub, col, anchor, ha) in enumerate(P):
        ax.scatter(xs[i], pr[i], s=34, color=col, edgecolor="white", linewidth=0.6, zorder=4)
        ax.annotate(f"{name}\n{sub}", xy=(xs[i], pr[i]), xytext=anchor, textcoords="data", fontsize=6.4,
                    color=INK, ha=ha, va="center", linespacing=1.2, zorder=5,
                    arrowprops=dict(arrowstyle="-", color=INK3, lw=0.5, shrinkA=2.0, shrinkB=3.5))
    ax.set_xlim(0.06, 1.10)
    ax.set_ylim(-1.5, 36)
    ax.set_xlabel("Rater agreement index $|V\\geq N|\\,/\\,|V\\geq1|$")
    ax.set_ylabel("Protocol effect (Dice points, optimal threshold)")
    ax.text(0.99, 0.99, f"Spearman ρ = {num(rho)}\nexact permutation p = {pval:.3f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.8, color=INK2)
    grid(ax)
    panel_tag(ax, "a", x=-0.12)

    order = np.argsort(xs)[::-1]
    yy = np.arange(len(order))
    h = 0.36
    bx.barh(yy - h / 2, pr[order], h, color=BLUE, label="Evaluation target (protocol)", zorder=3)
    bx.barh(yy + h / 2, su[order], h, color=ORANGE, label="Supervision target (learning design)", zorder=3)
    for y_, v in zip(yy - h / 2, pr[order]):
        bx.text(v + 0.5, y_, f"{v:.1f}", va="center", fontsize=6.4, fontweight="bold")
    for y_, v in zip(yy + h / 2, su[order]):
        bx.text(v + 0.5, y_, f"{v:.1f}", va="center", fontsize=6.4, fontweight="bold")
    bx.set_yticks(yy, [f"{P[i][1]}  ({xs[i]:.2f})" for i in order], fontsize=6.7)
    bx.invert_yaxis()
    bx.set_ylim(len(order) - 0.5, -0.55)
    bx.set_xlim(0, 36)
    bx.set_xlabel("Effect size (Dice points)")
    bx.tick_params(axis="y", length=0, colors=INK)
    bx.spines["left"].set_visible(False)
    grid(bx, "x")
    bx.legend(loc="lower right", bbox_to_anchor=(1.0, 0.995), fontsize=6.6, handlelength=1.0,
              handleheight=0.9, labelspacing=0.3)
    panel_tag(bx, "b", x=-0.66)
    VALUES["fig9"] = {"spearman_rho": round(rho, 4), "exact_p": round(pval, 4),
                      "points": {k: {"agreement": round(pts[k][0], 3), "protocol": round(pts[k][1], 2),
                                     "supervision": round(pts[k][2], 2)} for k in keys}}
    save(fig, "Fig9")


if __name__ == "__main__":
    which = sys.argv[1:] or [f"fig{i}" for i in range(1, 10)]
    for name in which:
        globals()[name]()
    vfile = os.path.join(OUT, "figure_values_v22.json")
    old = {}
    if os.path.exists(vfile) and len(which) < 9:
        old = json.load(open(vfile, encoding="utf-8"))
    old.update(VALUES)
    with open(vfile, "w", encoding="utf-8") as f:
        json.dump(old, f, ensure_ascii=False, indent=1)
    print(f"图上数值 -> {vfile}")
