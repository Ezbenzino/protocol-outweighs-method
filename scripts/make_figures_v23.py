# -*- coding: utf-8 -*-
"""make_figures_v23.py — all nine figures of manuscript v23, regenerated from released data (2026-10-11).

Inputs (all real experiment outputs in this repository):
  outputs/figures/v23/source_data/*      case-level tables written by scripts/make_figure_source_data_v23.py
  outputs/analysis_full/symmetric_analysis.json, noise_floor.json, qubiq_family_sensitivity.json
  outputs/analysis_test/symmetric_analysis_applied.json
  outputs/analysis/effect_uncertainty.json, case_lesion_sensitivity.json, disagreement_index.json,
                   qubiq_all_effects.json, generalisation_table.json
  outputs/analysis_scope/offset_paired_stats.json, diag_offset.json, diag_offset_allarms.json, scope_analysis_merged2.json
  outputs/figures/v22/source_data/cv_threshold_curves.csv, cv_reference_areas.json
Outputs: outputs/figures/v23/Fig1..Fig9 (.png 600 dpi, .pdf vector) and figure_values_v23.json
Usage (repository root):
    python scripts/make_figure_source_data_v23.py
    python scripts/make_figures_v23.py            # or: python scripts/make_figures_v23.py fig3 fig5
"""
import json
import os

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects as pe

ROOT = os.environ.get("FIG_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root (script lives in scripts/)
OUT = os.environ.get("FIG_OUT") or os.path.join(ROOT, "outputs", "figures", "v23")
SD = os.path.join(ROOT, "outputs", "figures", "v23", "source_data")
os.makedirs(OUT, exist_ok=True)

W = 6.5                       # text width in inches (= 16.51 cm embed width in the Word manuscript)
BLUE, ORANGE, VIOLET, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#4a3aa7", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, INK3, GRID, BAND = "#22221f", "#52514e", "#8a8983", "#e6e5e0", "#efeee9"
GREY = "#b9b8b1"
NOISE_C = "#8f8f8a"
ARM_C = {"tgtA": BLUE, "tgtB": ORANGE, "tgtC": VIOLET, "tgtD": AQUA, "tgtE": MAGENTA, "tgtF": YELLOW}
ARM_L = {"tgtA": "A union", "tgtB": "B majority", "tgtC": "C majority + soft", "tgtD": "D soft only",
         "tgtE": "E SVLS", "tgtF": "F SoftSeg", "archR18": "ResNet-18", "archCNX": "ConvNeXt-T",
         "archPVT": "PVT v2-b1", "archPU1e3": "Plain U-Net", "tgtBshift": "B + translation",
         "bg_maj": "B + background", "tgtBshift_bg": "B + both"}
CAT_C = {"protocol": BLUE, "data": VIOLET, "method": ORANGE, "noise": NOISE_C}
# ordinal ramp for the nested references G1..G4 (magma steps; same map as the vote-map image)
_mag = plt.get_cmap("magma")
REF_C = [mpl.colors.to_hex(_mag(x)) for x in (0.86, 0.66, 0.45, 0.24)]
VOTE_CMAP = mpl.colors.ListedColormap(["#000000"] + [mpl.colors.to_hex(_mag(x)) for x in (0.24, 0.45, 0.66, 0.86)])
MINUS = "−"

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic",
    "mathtext.bf": "Liberation Sans:bold",
    "font.size": 6.8, "axes.labelsize": 6.8, "axes.titlesize": 7.0,
    "xtick.labelsize": 6.3, "ytick.labelsize": 6.3, "legend.fontsize": 6.3,
    "axes.linewidth": 0.55, "xtick.major.width": 0.55, "ytick.major.width": 0.55,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2, "xtick.minor.size": 1.3, "ytick.minor.size": 1.3,
    "xtick.major.pad": 1.8, "ytick.major.pad": 1.8, "axes.labelpad": 2.0, "axes.titlepad": 3.0,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK3,
    "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK, "text.color": INK,
    "legend.frameon": False, "legend.handlelength": 1.4, "legend.borderaxespad": 0.2,
    "figure.dpi": 120, "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": True,
})

VALUES = {}


def J(*p):
    with open(os.path.join(ROOT, *p), encoding="utf-8") as f:
        return json.load(f)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  -> {name}")


def grid(ax, axis="y"):
    ax.grid(True, axis=axis, color=GRID, lw=0.45, zorder=0)
    ax.set_axisbelow(True)


def tag(ax, s, x=-0.16, y=1.02, fig=None):
    """Bold lowercase panel label (Nature style)."""
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="bottom", ha="left", color=INK)


def ftag(fig, s, x, y):
    fig.text(x, y, s, fontsize=8.5, fontweight="bold", va="top", ha="left", color=INK)


def num(x, nd=2):
    return f"{x:.{nd}f}".replace("-", MINUS)


def sgn(x, nd=2):
    return f"{x:+.{nd}f}".replace("-", MINUS)


def halo(lw=1.6, c="white"):
    return [pe.Stroke(linewidth=lw, foreground=c), pe.Normal()]


def boot_mean_ci(x, B=5000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)])
    return np.percentile(m, [2.5, 97.5])


# ------------------------------------------------------------------ loaders
_cache = {}


def case_cv():
    if "cv" not in _cache:
        _cache["cv"] = pd.read_csv(os.path.join(SD, "case_cv.csv.gz"))
    return _cache["cv"]


def case_test():
    if "te" not in _cache:
        _cache["te"] = pd.read_csv(os.path.join(SD, "case_test.csv.gz"))
    return _cache["te"]


def scope_case():
    if "sc" not in _cache:
        _cache["sc"] = pd.read_csv(os.path.join(SD, "scope_case.csv.gz"))
    return _cache["sc"]


def thr_cv():
    return J("outputs", "analysis_full", "symmetric_analysis.json")["table1"]


def pc_cv(arm, k, thr):
    d = case_cv()
    s = d[(d.arm == arm) & (d.k == k) & np.isclose(d.threshold, thr)]
    return s.set_index(["fold", "case_id"]).dice


def pc_test(arm, k, thr, stratum="all"):
    d = case_test()
    s = d[(d.arm == arm) & (d.k == k) & np.isclose(d.threshold, thr) & (d.stratum == stratum)]
    return s.set_index("case_id").dice


def cv_value(arm, k, thr):
    """CV estimate exactly as analyze_symmetric.py: mean over folds of case means."""
    return pc_cv(arm, k, thr).groupby(level=0).mean().mean()


def logit_axis(ax, lim=(0.0008, 0.9992), majors=(0.01, 0.5, 0.99)):
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    ax.set_xscale("logit")
    ax.xaxis.set_major_locator(FixedLocator(list(majors)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(FixedLocator([0.002, 0.005, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7,
                                             0.8, 0.9, 0.95, 0.98, 0.995, 0.998]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(*lim)

# ====================================================================== Fig. 1
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch


def _box(fig, x, y, w, h, fc="#f6f6f3", ec=INK3, lw=0.5, r=0.006):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", transform=fig.transFigure,
                       fc=fc, ec=ec, lw=lw, zorder=-1)
    fig.add_artist(p)
    return p


def _arrow(fig, x0, y0, x1, y1, c=INK2, lw=0.7, ms=6):
    fig.add_artist(FancyArrowPatch((x0, y0), (x1, y1), transform=fig.transFigure, arrowstyle="-|>",
                                   mutation_scale=ms, color=c, lw=lw, shrinkA=0, shrinkB=0))


def fig1():
    z = np.load(os.path.join(SD, "gallery_candidates.npz"))
    meta = json.load(open(os.path.join(SD, "gallery_candidates.json")))
    gi = 30                                   # 8.1 mm spiculated nodule, LIDC-IDRI-0608 (cross-validation pool)
    img, vote = z["imgs"][gi].astype(float), z["votes"][gi]
    c = 32; h = 22
    img, vote = img[c - h:c + h, c - h:c + h], vote[c - h:c + h, c - h:c + h]
    fig = plt.figure(figsize=(W, 4.75))

    # ======================= a: from four readings to nested references
    ftag(fig, "a", 0.0, 0.995)
    fig.text(0.03, 0.992, "Reference standard is derived, not given: one vote map, four nested references",
             fontsize=6.6, fontweight="bold", va="top")
    y0, hh = 0.775, 0.165
    ww = hh * 4.75 / W
    xs = [0.03, 0.03 + ww + 0.035]
    ax = fig.add_axes([xs[0], y0, ww, hh]); ax.imshow(img, cmap="gray", vmin=0, vmax=255); ax.axis("off")
    ax.set_title("CT, lung window", fontsize=5.8, pad=2)
    ax = fig.add_axes([xs[1], y0, ww, hh]); ax.imshow(vote, cmap=VOTE_CMAP, vmin=-0.5, vmax=4.5, interpolation="nearest"); ax.axis("off")
    ax.set_title("vote map V(x) = 0 … 4", fontsize=5.8, pad=2)
    _arrow(fig, xs[0] + ww + 0.006, y0 + hh / 2, xs[1] - 0.006, y0 + hh / 2)
    fig.text((xs[0] + ww + xs[1]) / 2, y0 + hh / 2 + 0.03, "4\nreaders", ha="center", fontsize=5.0, color=INK2, linespacing=1.0)
    x = xs[1] + ww + 0.035
    _arrow(fig, xs[1] + ww + 0.006, y0 + hh / 2, x - 0.006, y0 + hh / 2)
    for k in (1, 2, 3, 4):
        ax = fig.add_axes([x, y0, ww, hh])
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        m = (vote >= k).astype(float)
        ax.imshow(np.ma.masked_where(m == 0, m), cmap=mpl.colors.ListedColormap([REF_C[k - 1]]), alpha=0.55, vmin=0, vmax=1)
        ax.contour(m, [0.5], colors=[REF_C[k - 1]], linewidths=0.8)
        ax.axis("off")
        ax.set_title(f"G$_{k}$: V ≥ {k}  ({int(m.sum())} px)", fontsize=5.8, pad=2)
        if k == 1:
            gx0 = x
        x += ww + 0.012
    gx1 = x - 0.012
    ax = fig.add_axes([x + 0.015, y0, ww, hh])
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    p = np.ma.masked_where(vote == 0, vote / 4.0)
    ax.imshow(p, cmap="magma", vmin=0, vmax=1.05, alpha=0.8)
    ax.axis("off")
    ax.set_title("p = V/4 (soft target)", fontsize=5.8, pad=2)
    fig.text((gx0 + gx1) / 2, y0 - 0.008, "nested by construction: G\u2081 \u2287 G\u2082 \u2287 G\u2083 \u2287 G\u2084  (union, majority, high consensus, unanimous)",
             fontsize=5.0, color=INK2, va="top", ha="center", fontfamily="DejaVu Sans")

    # ======================= b: design matrix
    ftag(fig, "b", 0.0, 0.715)
    fig.text(0.03, 0.712, "Controlled design: arms differ from base arm B only where marked",
             fontsize=6.6, fontweight="bold", va="top")
    cols = ["Dice term", "Pixel term", "Encoder", "Transl.", "Backgr.", "Runs", "CV", "Test", "Offset", "FOV"]
    cx = [0.155, 0.215, 0.29, 0.355, 0.398, 0.44, 0.478, 0.512, 0.548, 0.585]
    rows = [
        ("Supervision target", None),
        ("A  union", ["G₁", "G₁", "R34", "", "", "5", 1, 1, 1, 0]),
        ("B  majority (base)", ["G₂", "G₂", "R34", "", "", "5", 1, 1, 1, 1]),
        ("C  majority + soft", ["G₂", "p", "R34", "", "", "5", 1, 1, 1, 0]),
        ("D  soft only", ["p", "p", "R34", "", "", "5", 1, 1, 1, 0]),
        ("E  SVLS", ["G₂", "SVLS(G₂)", "R34", "", "", "5", 1, 1, 1, 0]),
        ("F  SoftSeg*", ["—", "p (AW)", "R34", "", "", "5", 1, 1, 1, 0]),
        ("Architecture (target B)", None),
        ("ResNet-18", ["G₂", "G₂", "R18", "", "", "5", 1, 1, 1, 0]),
        ("ConvNeXt-Tiny", ["G₂", "G₂", "CNX-T", "", "", "5", 1, 1, 1, 0]),
        ("PVT v2-b1", ["G₂", "G₂", "PVT", "", "", "5", 1, 1, 1, 0]),
        ("Plain U-Net", ["G₂", "G₂", "none", "", "", "5", 1, 1, 1, 0]),
        ("Data protocol (target B)", None),
        ("+ translation ±32 px", ["G₂", "G₂", "R34", "●", "", "5", 1, 1, 1, 1]),
        ("+ background 50%", ["G₂", "G₂", "R34", "", "●", "2", 1, 1, 0, 1]),
        ("+ both", ["G₂", "G₂", "R34", "●", "●", "2", 1, 1, 0, 1]),
        ("Noise floor", None),
        ("A, B × 3 seeds", ["", "", "R34", "", "", "12", 1, 0, 0, 0]),
    ]
    ytop, dy = 0.655, 0.0318
    for xi, cname in zip(cx, cols):
        fig.text(xi, ytop + 0.006, cname, ha="center", va="bottom", fontsize=5.1, color=INK2, rotation=0)
    fig.add_artist(mpl.lines.Line2D([0.03, 0.605], [ytop + 0.002, ytop + 0.002], transform=fig.transFigure, color=INK3, lw=0.5))
    y = ytop - 0.016
    for name, vals in rows:
        if vals is None:
            fig.text(0.03, y, name, fontsize=5.5, fontweight="bold", va="center", color=INK)
            y -= dy * 0.92
            continue
        base = name.startswith("B  majority")
        if base:
            fig.add_artist(Rectangle((0.03, y - dy / 2), 0.575, dy, transform=fig.transFigure, fc="#fdeee6", ec="none", zorder=0))
        fig.text(0.04, y, name, fontsize=5.4, va="center", color=INK)
        for j, (xi, v) in enumerate(zip(cx, vals)):
            if j >= 6:
                fig.add_artist(mpl.lines.Line2D([xi], [y], transform=fig.transFigure, marker="o", ls="none",
                                                ms=3.3 if v else 2.6, mfc=INK2 if v else "white", mec=INK2 if v else GREY, mew=0.5))
            else:
                ref = ["G₂", "G₂", "R34", "", "", "5"][j]
                diff = (not base) and name != "A, B × 3 seeds" and j < 5 and v != ref
                if v == "●":
                    fig.add_artist(mpl.lines.Line2D([xi], [y], transform=fig.transFigure, marker="o", ls="none", ms=3.3,
                                                    mfc=ORANGE, mec=ORANGE))
                else:
                    fig.text(xi, y, v, fontsize=5.3, ha="center", va="center", color=ORANGE if diff else INK,
                             fontweight="bold" if diff else "normal")
        y -= dy
    fig.text(0.03, y + 0.004, "Orange: where an arm differs from base arm B. R34 = ImageNet ResNet-34 encoder; p = V/4;\nAW = Adaptive Wing loss. *F changes target, activation and loss together and is reported outside the factorial.\n"
             "71 LIDC training runs; all arms share optimiser, schedule, augmentation, splits and a Dice + BCE objective.", fontsize=4.9, color=INK2, va="top", linespacing=1.3)

    # ======================= c: evaluation expansion
    ftag(fig, "c", 0.635, 0.715)
    fig.text(0.665, 0.712, "One forward pass, every protocol", fontsize=6.6, fontweight="bold", va="top")
    _box(fig, 0.645, 0.395, 0.35, 0.29, fc="#fafaf8")
    # references
    fig.text(0.66, 0.655, "references", fontsize=5.5, color=INK2, va="center")
    for k in (1, 2, 3, 4):
        fig.add_artist(mpl.patches.Circle((0.765 + (k - 1) * 0.004, 0.655), 0.017 - (k - 1) * 0.0035, transform=fig.transFigure,
                                          fc=REF_C[k - 1], ec="white", lw=0.4))
    fig.text(0.80, 0.655, "× 4  (V ≥ 1 … V ≥ 4)", fontsize=5.5, va="center")
    # thresholds
    fig.text(0.66, 0.60, "thresholds", fontsize=5.5, color=INK2, va="center")
    tax = fig.add_axes([0.745, 0.592, 0.12, 0.018]); tax.set_xscale("logit")
    tt = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75,
          0.8, 0.85, 0.9, 0.925, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998, 0.999]
    tax.vlines(tt, 0, 1, color=INK2, lw=0.35); tax.set_xlim(0.0008, 0.9992); tax.axis("off")
    fig.text(0.87, 0.60, "× 33, 0.001–0.999", fontsize=5.5, va="center")
    fig.text(0.66, 0.548, "operating\npoints", fontsize=5.5, color=INK2, va="center", linespacing=1.1)
    fig.text(0.745, 0.548, "fixed 0.5 · calibrated on validation\n· area-matched to G$_2$", fontsize=5.3, va="center", linespacing=1.2)
    # scope ladder
    fig.text(0.66, 0.475, "scope", fontsize=5.5, color=INK2, va="center")
    for i, s in enumerate((0.012, 0.018, 0.024, 0.036, 0.048)):
        fig.add_artist(Rectangle((0.775 - s / 2, 0.475 - s * W / 4.75 / 2), s, s * W / 4.75, transform=fig.transFigure, fill=False,
                                 ec=["#0d366b", "#1c5cab", "#3987e5", "#6da7ec", "#9ec5f4"][i], lw=0.6))
    fig.text(0.81, 0.475, "patch 128 · windows 128–512 ·\nslice · volume  (× 8)", fontsize=5.3, va="center", linespacing=1.2)
    # offsets
    fig.text(0.66, 0.418, "offset", fontsize=5.5, color=INK2, va="center")
    for dd, a_ in ((0, 1.0), (0.008, 0.7), (0.016, 0.45)):
        fig.add_artist(Rectangle((0.758 + dd, 0.405), 0.024, 0.024 * W / 4.75, transform=fig.transFigure, fill=False, ec=ORANGE, lw=0.6, alpha=a_))
    fig.text(0.81, 0.418, "d = 0–56 px × 6 directions", fontsize=5.3, va="center")

    # ======================= d: cohorts
    ftag(fig, "d", 0.635, 0.375)
    fig.text(0.665, 0.372, "Data, splits and replication", fontsize=6.6, fontweight="bold", va="top")
    _box(fig, 0.645, 0.275, 0.35, 0.075, fc="#eef4fc", ec="#9ec5f4")
    fig.text(0.655, 0.312, "LIDC-IDRI: 1018 series parsed, 1010 enter the study;\ncase-level split; lesion = V ≥ 2 component ≥ 3 mm", fontsize=5.3, va="center", linespacing=1.25)
    _box(fig, 0.645, 0.155, 0.168, 0.095, fc="white")
    fig.text(0.729, 0.203, "Cross-validation\n808 cases, 5 folds\n634 contributing\n8471 instances", ha="center", va="center", fontsize=5.2, linespacing=1.25)
    _box(fig, 0.827, 0.155, 0.168, 0.095, fc="white", ec=ORANGE)
    fig.text(0.911, 0.203, "Held-out test\n202 cases, never tuned\n158 contributing\n2062 instances", ha="center", va="center", fontsize=5.2, linespacing=1.25)
    _arrow(fig, 0.73, 0.275, 0.73, 0.252); _arrow(fig, 0.91, 0.275, 0.91, 0.252)
    _arrow(fig, 0.815, 0.18, 0.825, 0.18, c=ORANGE)
    fig.text(0.82, 0.145, "thresholds\nimported", ha="center", va="top", fontsize=4.6, color=ORANGE, linespacing=1.1)
    _box(fig, 0.645, 0.03, 0.35, 0.085, fc="#f3f1fb", ec="#b7b0e6")
    fig.text(0.655, 0.0725, "External: QUBIQ 2021, 7 tasks from 4 collections\n(brain growth, brain tumour ×3, kidney, prostate ×2;\n"
             "3–7 raters; CT and MRI) × 3 targets × 5 folds = 105 runs", fontsize=5.1, va="center", linespacing=1.25)
    save(fig, "Fig1")



# ====================================================================== Fig. 2


GALLERY = [4, 18, 29, 30, 44, 50, 57, 67]     # indices into gallery_candidates.npz (chosen by eye for clarity)


def fig2():
    z = np.load(os.path.join(SD, "gallery_candidates.npz"))
    meta = json.load(open(os.path.join(SD, "gallery_candidates.json")))
    ic = pd.read_csv(os.path.join(SD, "instances_cv.csv")); ic["set"] = "Cross-validation"
    it = pd.read_csv(os.path.join(SD, "instances_test.csv")); it["set"] = "Held-out test"
    inst = pd.concat([ic, it], ignore_index=True)
    for k in (2, 3, 4):
        inst[f"r{k}"] = inst[f"gt_area_v{k}"] / inst.gt_area_v1
    inst["stratum"] = np.where(inst.diameter_mm < 5, "3–5", np.where(inst.diameter_mm < 10, "5–10", "10–30"))

    fig = plt.figure(figsize=(W, 3.85))
    # ---------------- a: gallery
    gs_top = fig.add_gridspec(1, 8, left=0.0, right=1.0, top=0.895, bottom=0.695, wspace=0.05)
    groups = {"3–5 mm": (0, 1), "5–10 mm": (2, 4), "10–30 mm": (5, 7)}
    axes = []
    for j, gi in enumerate(GALLERY):
        ax = fig.add_subplot(gs_top[0, j]); axes.append(ax)
        im, v = z["imgs"][gi].astype(float), z["votes"][gi]
        h = 22 if j < 2 else (26 if j < 5 else 32)            # zoom: crops shrink with lesion size
        c = im.shape[0] // 2
        im, v = im[c - h:c + h, c - h:c + h], v[c - h:c + h, c - h:c + h]
        ax.imshow(im, cmap="gray", vmin=0, vmax=255, interpolation="bilinear")
        for k in (1, 2, 3, 4):
            m = (v >= k).astype(float)
            if m.sum():
                cs = ax.contour(m, [0.5], colors=[REF_C[k - 1]], linewidths=0.85)
                for col in cs.collections if hasattr(cs, "collections") else [cs]:
                    col.set_path_effects(halo(1.7, "#000000"))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color("#c9c8c2"); sp.set_linewidth(0.5)
        a = meta[gi]["a"]
        ratio = a[3] / a[0]
        ax.text(0.04, 0.96, f"{meta[gi]['diam']:.1f} mm", transform=ax.transAxes, color="white", fontsize=5.8,
                va="top", ha="left", path_effects=halo(1.4, "#000000"))
        lab = "no unanimous pixel" if a[3] == 0 else f"|G₄|/|G₁| = {ratio:.2f}"
        ax.text(0.5, -0.05, f"{a[0]}/{a[1]}/{a[2]}/{a[3]} px\n{lab}", transform=ax.transAxes, ha="center",
                va="top", fontsize=5.5, color=INK2, linespacing=1.15)
    for name, (j0, j1) in groups.items():
        x0 = axes[j0].get_position().x0; x1 = axes[j1].get_position().x1
        fig.add_artist(mpl.lines.Line2D([x0 + 0.004, x1 - 0.004], [0.905, 0.905], color=INK3, lw=0.6))
        fig.text((x0 + x1) / 2, 0.912, name, ha="center", va="bottom", fontsize=6.3, color=INK2)
    ftag(fig, "a", -0.005, 0.995)
    # legend for the contours
    hx = 0.60
    for k in (1, 2, 3, 4):
        fig.add_artist(mpl.lines.Line2D([hx + (k - 1) * 0.09, hx + (k - 1) * 0.09 + 0.025], [0.988, 0.988],
                                        color=REF_C[k - 1], lw=1.6))
        fig.text(hx + (k - 1) * 0.09 + 0.03, 0.988, f"V ≥ {k}", va="center", fontsize=6.0, color=INK2)
    VALUES["fig2_gallery"] = [dict(meta[g], shown=True) for g in GALLERY]

    # ---------------- b-e: cohort statistics
    gs = fig.add_gridspec(1, 4, left=0.07, right=0.995, top=0.50, bottom=0.105, wspace=0.55,
                          width_ratios=[1.05, 1.0, 1.15, 0.85])
    # b: ECDF of |Gk|/|G1|
    ax = fig.add_subplot(gs[0, 0])
    for k in (2, 3, 4):
        x = np.sort(inst[f"r{k}"].values)
        y = np.arange(1, len(x) + 1) / len(x)
        ax.step(x, y, where="post", color=REF_C[k - 1], lw=1.2, label=f"V ≥ {k}")
    p4 = (inst.r4 == 0).mean(); p3 = (inst.r3 == 0).mean()
    ax.annotate(f"{100*p4:.1f}% of instances\nhave no unanimous pixel", xy=(0.005, p4), xytext=(0.06, 0.70),
                fontsize=5.6, color=INK2, bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.9),
                arrowprops=dict(arrowstyle="-", lw=0.5, color=INK3))
    ax.set_xlim(-0.02, 1.0); ax.set_ylim(0, 1.0)
    ax.set_xlabel("Reference area relative to union, |G$_k$|/|G$_1$|")
    ax.set_ylabel("Cumulative fraction of instances")
    ax.legend(loc="lower right", fontsize=5.8, handlelength=1.2)
    grid(ax, "both")
    tag(ax, "b", x=-0.30, y=1.03)

    # c: empty G3/G4 by stratum, CV vs test
    bx = fig.add_subplot(gs[0, 1])
    strata = ["3–5", "5–10", "10–30"]
    xs = np.arange(3)
    w = 0.19
    out_c = {}
    for i, (k, setname, hatch) in enumerate([(3, "Cross-validation", None), (3, "Held-out test", "////"),
                                             (4, "Cross-validation", None), (4, "Held-out test", "////")]):
        vals = []
        for st in strata:
            s = inst[(inst.stratum == st) & (inst.set == setname)]
            vals.append(100 * (s[f"gt_area_v{k}"] == 0).mean())
        off = (i - 1.5) * (w + 0.02)
        bx.bar(xs + off, vals, w, color=REF_C[k - 1], edgecolor="white" if hatch is None else REF_C[k - 1],
               hatch=hatch, lw=0.0 if hatch is None else 0.0, zorder=3, fill=hatch is None)
        if hatch:
            bx.bar(xs + off, vals, w, facecolor="white", edgecolor=REF_C[k - 1], hatch="//////", lw=0.6, zorder=3)
        out_c[f"G{k}_{setname}"] = [round(v, 1) for v in vals]
    n_by = [int((inst.stratum == st).sum()) for st in strata]
    bx.set_xticks(xs, [f"{s}\n({n})" for s, n in zip(strata, n_by)], fontsize=5.8)
    bx.set_xlabel("Diameter stratum, mm (instances)")
    bx.set_ylabel("Instances with empty reference (%)")
    bx.set_ylim(0, 100)
    grid(bx)
    from matplotlib.patches import Patch
    bx.legend(handles=[Patch(facecolor=REF_C[2], label="V ≥ 3 empty"), Patch(facecolor=REF_C[3], label="V ≥ 4 empty"),
                       Patch(facecolor="white", edgecolor=INK3, hatch="//////", label="test (hatched)")],
              loc="upper right", fontsize=5.6, handlelength=1.0, handleheight=0.8)
    tag(bx, "c", x=-0.30, y=1.03)
    VALUES["fig2_empty_by_stratum_pct"] = out_c

    # d: relative reference area vs diameter (mean and 95% CI per diameter bin; empty references count as 0)
    cx = fig.add_subplot(gs[0, 2])
    d = inst.diameter_mm.values
    bins = np.array([3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 31])
    for k in (2, 3, 4):
        r = inst[f"r{k}"].values
        mids, mu, lo_, hi_ = [], [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (d >= lo) & (d < hi)
            mids.append(np.sqrt(lo * hi)); mu.append(r[m].mean())
            se = r[m].std(ddof=1) / np.sqrt(m.sum()); lo_.append(mu[-1] - 1.96 * se); hi_.append(mu[-1] + 1.96 * se)
        cx.fill_between(mids, lo_, hi_, color=REF_C[k - 1], alpha=0.25, lw=0, zorder=3)
        cx.plot(mids, mu, color=REF_C[k - 1], lw=1.3, marker="o", ms=2.6, mec="white", mew=0.4, zorder=4,
                label=f"V ≥ {k}")
        VALUES[f"fig2_mean_ratio_r{k}_by_diam_bin"] = {f"{lo}-{hi}": round(v, 3) for lo, hi, v in zip(bins[:-1], bins[1:], mu)}
    cx.set_xscale("log")
    cx.set_xticks([3, 5, 10, 20, 30], ["3", "5", "10", "20", "30"])
    cx.minorticks_off()
    cx.set_xlim(2.9, 31)
    cx.set_ylim(0, 1.0)
    cx.set_xlabel("Equivalent diameter (mm)")
    cx.set_ylabel("Mean |G$_k$|/|G$_1$|  (95% CI)")
    cx.legend(loc="lower right", fontsize=5.8, handlelength=1.2)
    grid(cx, "both")
    tag(cx, "d", x=-0.27, y=1.03)
    rho = pd.Series(d).corr(inst.r4, method="spearman")
    VALUES["fig2_spearman_diam_r4"] = round(float(rho), 3)

    # e: instances per case
    ex = fig.add_subplot(gs[0, 3])
    cnt = inst.groupby("case_id").size().values
    ex.hist(cnt, bins=np.arange(0, 82, 4), color=INK3, edgecolor="white", lw=0.4, zorder=3)
    ex.axvline(np.median(cnt), color=INK, lw=0.8, ls=(0, (3, 2)))
    ex.text(np.median(cnt) + 2, ex.get_ylim()[1] * 0.92, f"median {np.median(cnt):.0f}\nmax {cnt.max()}",
            fontsize=5.8, va="top", color=INK2)
    ex.set_xlabel("Instances per case")
    ex.set_ylabel("Cases (n = 792)")
    grid(ex)
    tag(ex, "e", x=-0.36, y=1.03)
    VALUES["fig2_instances_per_case"] = {"median": float(np.median(cnt)), "max": int(cnt.max()), "n_cases": int(len(cnt))}
    VALUES["fig2_pct_no_unanimous"] = {"instances": round(100 * p4, 1), "G3_empty_instances": round(100 * p3, 1)}
    save(fig, "Fig2")



# ====================================================================== Fig. 3


def fig3():
    T = thr_cv()
    cv = case_cv()
    curves_src = pd.read_csv(os.path.join(ROOT, "outputs", "figures", "v22", "source_data", "cv_threshold_curves.csv"))
    R = J("outputs", "figures", "v22", "source_data", "cv_reference_areas.json")
    T5 = J("outputs", "analysis_full", "symmetric_analysis.json")["table5"]
    Tt = J("outputs", "analysis_test", "symmetric_analysis_applied.json")["table1"]

    fig = plt.figure(figsize=(W, 5.55))
    # ---------------- a: threshold curves (CV), recomputed from per-case records
    gsa = fig.add_gridspec(1, 4, left=0.07, right=0.995, top=0.955, bottom=0.735, wspace=0.10)
    arms = ["tgtA", "tgtB", "tgtC", "tgtD"]
    g = cv[cv.arm.isin(arms)].groupby(["arm", "k", "threshold", "fold"]).dice.mean().groupby(["arm", "k", "threshold"]).mean()
    va = {}
    for j, k in enumerate((1, 2, 3, 4)):
        ax = fig.add_subplot(gsa[0, j])
        for a in arms:
            s = g.loc[(a, k)]
            ax.plot(s.index, s.values, color=ARM_C[a], lw=1.15, zorder=3)
            t = T[a][str(k)]["thr"]
            v = float(s.loc[np.isclose(s.index, t)].iloc[0])
            assert abs(v - T[a][str(k)]["dice_opt"]) < 1e-6
            ax.plot([t], [v], "o", ms=3.4, color=ARM_C[a], mec="white", mew=0.6, zorder=5)
            va[f"{a}_G{k}"] = {"thr": t, "dice": round(v, 4)}
        ax.axvline(0.5, color=INK3, lw=0.6, ls=(0, (3, 2)), zorder=1)
        logit_axis(ax)
        ax.set_ylim(0.15, 0.92)
        ax.set_title(f"Reference G$_{k}$ (V ≥ {k})", fontsize=6.8)
        grid(ax)
        if j:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("Dice (cross-validation)")
            tag(ax, "a", x=-0.30, y=1.04)

    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=ARM_C[a], lw=1.4, label=ARM_L[a]) for a in arms] + \
        [Line2D([], [], ls="none", marker="o", ms=3.4, color=INK3, mec="white", label="calibrated optimum"),
         Line2D([], [], color=INK3, lw=0.6, ls=(0, (3, 2)), label="fixed 0.5")]
    fig.text(0.535, 0.708, "Binarisation threshold (logit scale)", ha="center", va="top", fontsize=6.8)
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.535, 0.688), ncol=6, fontsize=6.0,
               handlelength=1.3, columnspacing=1.0)
    VALUES["fig3a"] = va

    # ---------------- b: cross-evaluation matrices (CV), fixed 0.5 and calibrated
    rows = ["tgtA", "tgtB", "tgtC", "tgtD", "tgtE"]
    gsb = fig.add_gridspec(1, 3, left=0.105, right=0.995, top=0.60, bottom=0.39, wspace=0.10,
                           width_ratios=[1, 1, 1.25])
    cmap = mpl.colors.LinearSegmentedColormap.from_list("b", ["#f4f8fd", "#9ec5f4", "#2a78d6", "#0d366b"])
    for j, (key, title) in enumerate([("dice_05", "Fixed threshold 0.5"), ("dice_opt", "Calibrated per arm and reference")]):
        ax = fig.add_subplot(gsb[0, j])
        M = np.array([[100 * T[a][str(k)][key] for k in (1, 2, 3, 4)] for a in rows])
        im = ax.imshow(M, cmap=cmap, vmin=58, vmax=88, aspect="auto")
        best = M.argmax(axis=0)
        for i in range(len(rows)):
            for c in range(4):
                dark = M[i, c] > 79
                ax.text(c, i, f"{M[i, c]:.1f}", ha="center", va="center", fontsize=5.7,
                        color="white" if dark else INK, fontweight="bold" if best[c] == i else "normal")
                if best[c] == i:
                    ax.add_patch(mpl.patches.Rectangle((c - 0.47, i - 0.45), 0.94, 0.9, fill=False, ec=INK, lw=0.7))
        ax.set_xticks(range(4), [f"G$_{k}$" for k in (1, 2, 3, 4)])
        ax.set_yticks(range(len(rows)), [ARM_L[a] for a in rows] if j == 0 else [])
        ax.tick_params(length=0, colors=INK)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title, fontsize=6.5)
        span = M.max(axis=0) - M.min(axis=0)
        ax.set_xlabel("within-column range: " + " / ".join(f"{v:.2f}" for v in span), fontsize=5.6, color=INK2)
        if j == 0:
            tag(ax, "b", x=-0.62, y=1.06)
        VALUES[f"fig3b_{key}"] = {a: [round(v, 2) for v in M[i]] for i, a in enumerate(rows)}
        VALUES[f"fig3b_{key}_column_range"] = [round(v, 2) for v in span]
        VALUES[f"fig3b_{key}_row_range"] = {a: round(float(M[i].max() - M[i].min()), 2) for i, a in enumerate(rows)}

    # ---------------- c: map of calibrated thresholds for every arm
    arms_t = ["tgtA", "tgtB", "tgtC", "tgtD", "tgtE", "archR18", "archCNX", "archPVT", "archPU1e3",
              "tgtBshift", "bg_maj", "tgtBshift_bg"]
    ax = fig.add_subplot(gsb[0, 2])
    Th = np.array([[T[a][str(k)]["thr"] for k in (1, 2, 3, 4)] for a in arms_t])
    L = np.log10(Th / (1 - Th))
    cm2 = mpl.colors.LinearSegmentedColormap.from_list("d", [BLUE, "#f0efec", "#e34948"])
    im2 = ax.imshow(L.T, cmap=cm2, vmin=-3, vmax=3, aspect="auto")
    for i in range(len(arms_t)):
        for c in range(4):
            t = Th[i, c]
            s = f"{t:g}".replace("0.", ".")
            ax.text(i, c, s, ha="center", va="center", fontsize=4.6, color="white" if abs(L[i, c]) > 1.9 else INK,
                    rotation=90)
    ax.set_yticks(range(4), [f"G$_{k}$" for k in (1, 2, 3, 4)])
    ax.set_xticks(range(len(arms_t)), ["A", "B", "C", "D", "E", "R18", "CNX", "PVT", "PU", "+tr", "+bg", "+both"],
                  rotation=90, fontsize=5.4)
    ax.tick_params(length=0, colors=INK)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Calibrated threshold, every arm", fontsize=6.5)
    tag(ax, "c", x=-0.15, y=1.06)
    cb2 = fig.colorbar(im2, ax=ax, fraction=0.05, pad=0.02, aspect=14, ticks=[-3, -2, -1, 0, 1, 2, 3])
    cb2.ax.set_yticklabels([".001", ".01", ".1", ".5", ".9", ".99", ".999"])
    cb2.ax.tick_params(labelsize=5.0, length=1.5, colors=INK2)
    cb2.outline.set_visible(False)
    VALUES["fig3c_thresholds"] = {a: list(map(float, Th[i])) for i, a in enumerate(arms_t)}

    # ---------------- d: per-case evaluation-target effect on the held-out test set
    gsd = fig.add_gridspec(1, 2, left=0.07, right=0.995, top=0.27, bottom=0.07, wspace=0.28, width_ratios=[1.35, 1])
    ax = fig.add_subplot(gsd[0, 0])
    items = [("A, G$_1$ − G$_4$\nfixed 0.5", "tgtA", 1, 4, "dice_05"), ("A, G$_1$ − G$_4$\ncalibrated", "tgtA", 1, 4, "dice_opt"),
             ("B, G$_2$ − G$_3$\nfixed 0.5", "tgtB", 2, 3, "dice_05"), ("B, G$_2$ − G$_3$\ncalibrated", "tgtB", 2, 3, "dice_opt")]
    rng = np.random.default_rng(3)
    out = {}
    for i, (lab, a, k1, k2, key) in enumerate(items):
        t1 = 0.5 if key == "dice_05" else Tt[a][str(k1)]["thr"]
        t2 = 0.5 if key == "dice_05" else Tt[a][str(k2)]["thr"]
        x1, x2 = pc_test(a, k1, t1), pc_test(a, k2, t2)
        x1, x2 = x1.align(x2, join="inner")
        dd = 100 * (x1 - x2).values
        col = BLUE if a == "tgtA" else ORANGE
        jit = rng.uniform(-0.18, 0.18, len(dd))
        ax.scatter(np.full(len(dd), i) + jit, dd, s=2.2, color=col, alpha=0.55, lw=0, zorder=3, rasterized=True)
        q1, med, q3 = np.percentile(dd, [25, 50, 75])
        ax.add_patch(mpl.patches.Rectangle((i + 0.24, q1), 0.13, q3 - q1, fc="white", ec=INK, lw=0.6, zorder=4))
        ax.plot([i + 0.24, i + 0.37], [med, med], color=INK, lw=0.9, zorder=5)
        ax.plot([i + 0.305], [dd.mean()], "D", ms=2.8, color=INK, zorder=6)
        pos = 100 * (dd > 0).mean()
        ax.text(i + 0.05, 104, f"{pos:.0f}% of cases > 0", ha="center", va="bottom", fontsize=5.3, color=INK2)
        out[lab.replace("\n", " ")] = {"n": int(len(dd)), "mean": round(dd.mean(), 2), "median": round(med, 2),
                                       "pct_positive": round(pos, 1)}
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks(range(len(items)), [it[0] for it in items], fontsize=5.6)
    ax.set_xlim(-0.5, len(items) - 0.4)
    ax.set_ylim(-45, 118)
    ax.set_ylabel("Per-case Dice change (points)")
    ax.set_title("Held-out test: one model, one set of predictions, two references", fontsize=6.5, pad=4)
    grid(ax)
    tag(ax, "d", x=-0.13, y=1.04)
    VALUES["fig3d_per_case"] = out

    # ---------------- e: predicted area vs threshold
    ex = fig.add_subplot(gsd[0, 1])
    for a in arms:
        s = curves_src[curves_src.arm == a].sort_values("threshold")
        ex.plot(s.threshold, s.area_case_mean, color=ARM_C[a], lw=1.15, zorder=3)
        t = float(T5[a]["threshold"])
        v = float(s.loc[np.isclose(s.threshold, t), "area_case_mean"].iloc[0])
        ex.plot([t], [v], "o", ms=3.2, color=ARM_C[a], mec="white", mew=0.6, zorder=5)
    for k in (1, 2, 3, 4):
        val = R["gt_area_case_mean_px"][f"G{k}"]
        ex.axhline(val, color=REF_C[k - 1], lw=0.9, ls=(0, (3, 1.5)), zorder=2)
        if k <= 2:
            ex.text(0.9985, val + 3, f"G$_{k}$ {val:.0f} px", ha="right", va="bottom", fontsize=5.3, color=INK2)
        else:
            ex.text(0.0009, val + 3, f"G$_{k}$ {val:.0f} px", ha="left", va="bottom", fontsize=5.3, color=INK2)
    logit_axis(ex)
    ex.set_ylim(0, 310)
    ex.set_xlabel("Binarisation threshold (logit scale)")
    ex.set_ylabel("Mean predicted area (px)")
    ex.set_title("Predicted extent vs reference extent (CV)", fontsize=6.5)
    grid(ex)
    tag(ex, "e", x=-0.22, y=1.04)
    save(fig, "Fig3")



# ====================================================================== Fig. 4
from scipy import stats


def _paired(a, b, k, test):
    if test:
        Tt = J("outputs", "analysis_test", "symmetric_analysis_applied.json")["table1"]
        x, y = pc_test(a, k, Tt[a][str(k)]["thr"]), pc_test(b, k, Tt[b][str(k)]["thr"])
    else:
        T = thr_cv()
        x, y = pc_cv(a, k, T[a][str(k)]["thr"]), pc_cv(b, k, T[b][str(k)]["thr"])
    x, y = x.align(y, join="inner")
    return 100 * (x - y).values


def fig4():
    noise = J("outputs", "analysis_full", "noise_floor.json")
    nf = noise["fixed_0_5"]["two_sigma_pts"]
    fig = plt.figure(figsize=(W, 4.45))
    gs = fig.add_gridspec(2, 2, left=0.115, right=0.995, top=0.955, bottom=0.115, wspace=0.30, hspace=0.42,
                          width_ratios=[1.0, 1.15])
    # ---------------- a: forest plot, CV and held-out test
    ax = fig.add_subplot(gs[:, 0])
    contrasts = [("tgtC", "tgtA", "C − A  conventional comparison", BLUE),
                 ("tgtB", "tgtA", "B − A  mask extent only", ORANGE),
                 ("tgtC", "tgtB", "C − B  soft supervision only", VIOLET),
                 ("tgtD", "tgtB", "D − B  soft only vs hard", AQUA),
                 ("tgtC", "tgtE", "C − E  rater votes vs Gaussian kernel", MAGENTA)]
    y = 0.0
    yt, yl, out = [], [], {}
    for a, b, title, col in contrasts:
        ax.text(-5.35, y + 0.15, title, fontsize=5.9, fontweight="bold", va="center", ha="left", color=INK)
        y -= 0.75
        for k in (1, 2, 3, 4):
            for test, dy, mk in ((False, 0.13, "o"), (True, -0.13, "D")):
                d = _paired(a, b, k, test)
                lo, hi = np.percentile([d[np.random.default_rng(0).integers(0, len(d), len(d))].mean()
                                        for _ in range(1)], [0, 100]) if False else boot_mean_ci(d, B=3000)
                m = d.mean()
                ax.plot([lo, hi], [y + dy] * 2, color=col, lw=1.1 if not test else 0.8, alpha=1 if not test else 0.75,
                        solid_capstyle="round", zorder=4)
                ax.plot([m], [y + dy], mk, ms=3.6 if not test else 3.0, color=col if not test else "white",
                        mec=col, mew=0.8, zorder=5)
                out[f"{a}-{b}_G{k}_{'test' if test else 'cv'}"] = {"n": int(len(d)), "diff": round(m, 2),
                                                                    "ci": [round(lo, 2), round(hi, 2)]}
            yt.append(y); yl.append(f"G$_{k}$")
            y -= 0.55
        y -= 0.25
    ax.axvspan(-nf, nf, color=BAND, zorder=0, lw=0)
    ax.axvline(0, color=INK2, lw=0.6, zorder=2)
    ax.set_yticks(yt, yl, fontsize=6.0)
    ax.tick_params(axis="y", length=0, colors=INK)
    ax.set_xlim(-5.4, 2.6)
    ax.set_ylim(y + 0.3, 0.5)
    ax.spines["left"].set_visible(False)
    grid(ax, "x")
    ax.set_xlabel("Paired Dice difference (points), case level")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="-", color=INK2, mec=INK2, ms=3.4, lw=1.0, label="cross-validation (n ≤ 634)"),
                       Line2D([], [], marker="D", ls="-", color=INK2, mfc="white", mec=INK2, ms=3.0, lw=0.8, label="held-out test (n ≤ 158)"),
                       mpl.patches.Patch(color=BAND, label=f"noise floor ±{nf:.2f} (2σ)")],
              loc="upper center", bbox_to_anchor=(0.45, -0.075), ncol=2, fontsize=5.6, handlelength=1.6, columnspacing=1.0)
    tag(ax, "a", x=-0.20, y=1.0)
    VALUES["fig4a"] = out

    # ---------------- b: per-case distribution of C − B (CV)
    bx = fig.add_subplot(gs[0, 1])
    data, labs = [], []
    for k in (1, 2, 3, 4):
        d = _paired("tgtC", "tgtB", k, False)
        data.append(d); labs.append(k)
    parts = bx.violinplot(data, positions=range(4), widths=0.8, showextrema=False, bw_method=0.25)
    for pc_ in parts["bodies"]:
        pc_.set_facecolor(VIOLET); pc_.set_alpha(0.22); pc_.set_edgecolor("none")
    rng = np.random.default_rng(1)
    vb = {}
    for i, d in enumerate(data):
        bx.scatter(i + rng.uniform(-0.09, 0.09, len(d)), d, s=0.9, color=VIOLET, alpha=0.45, lw=0, rasterized=True, zorder=3)
        q1, med, q3 = np.percentile(d, [25, 50, 75])
        bx.plot([i - 0.16, i + 0.16], [med, med], color=INK, lw=0.9, zorder=5)
        bx.plot([i, i], [q1, q3], color=INK, lw=2.2, zorder=4, solid_capstyle="butt")
        bx.plot([i], [d.mean()], "D", ms=2.8, color="white", mec=INK, mew=0.7, zorder=6)
        b_better = 100 * (d < 0).mean()
        bx.text(i, 19.5, f"n = {len(d)}\nB better {b_better:.0f}%", ha="center", va="top", fontsize=5.2, color=INK2,
                linespacing=1.1)
        vb[f"G{i+1}"] = {"n": int(len(d)), "mean": round(d.mean(), 2), "median": round(med, 2), "pct_B_better": round(b_better, 1)}
    bx.axhline(0, color=INK2, lw=0.6)
    bx.set_ylim(-16, 20)
    bx.set_yticks([-15, -10, -5, 0, 5, 10, 15])
    bx.set_xticks(range(4), [f"G$_{k}$" for k in (1, 2, 3, 4)])
    bx.set_ylabel("C − B per case (points)")
    bx.set_title("Soft supervision, case by case (cross-validation)", fontsize=6.4)
    grid(bx)
    tag(bx, "b", x=-0.17, y=1.02)
    VALUES["fig4b"] = vb

    # ---------------- c: equivalence (TOST) and d: noise floor
    gsr = gs[1, 1].subgridspec(1, 2, wspace=0.55, width_ratios=[1, 1.1])
    cx = fig.add_subplot(gsr[0, 0])
    rows = [("G$_1$ CV", 1, False), ("G$_2$ CV", 2, False), ("G$_1$ test", 1, True), ("G$_2$ test", 2, True)]
    for b_, a_ in ((1.5, 0.10), (nf, 0.18), (1.0, 0.28)):
        cx.axvspan(-b_, b_, color=NOISE_C, alpha=a_ * 0.6, lw=0, zorder=0)
    tv = {}
    for i, (lab, k, test) in enumerate(rows):
        d = _paired("tgtC", "tgtB", k, test)
        m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
        t = stats.t.ppf(0.95, len(d) - 1)
        lo, hi = m - t * se, m + t * se
        yy = len(rows) - 1 - i
        cx.plot([lo, hi], [yy, yy], color=VIOLET, lw=1.3, zorder=4, solid_capstyle="round")
        cx.plot([m], [yy], "o" if not test else "D", ms=3.4, color=VIOLET if not test else "white", mec=VIOLET, mew=0.8, zorder=5)
        p_tost = max(stats.t.sf((m + 1.0) / se, len(d) - 1), stats.t.cdf((m - 1.0) / se, len(d) - 1))
        tv[lab] = {"mean": round(m, 3), "ci90": [round(lo, 2), round(hi, 2)], "p_tost_1.0": float(p_tost)}
    cx.axvline(0, color=INK2, lw=0.6)
    cx.set_yticks(range(len(rows)), [r[0] for r in rows][::-1], fontsize=5.8)
    cx.tick_params(axis="y", length=0)
    cx.set_xlim(-1.7, 1.7)
    cx.set_xticks([-1.5, -1.0, 0, 1.0, 1.5], ["−1.5", "−1", "0", "1", "1.5"])
    cx.set_xlabel("C − B, 90% CI (points)")
    cx.set_title("Equivalence (TOST)", fontsize=6.0)
    cx.spines["left"].set_visible(False)
    tag(cx, "c", x=-0.42, y=1.04)
    VALUES["fig4c"] = tv

    dx = fig.add_subplot(gsr[0, 1])
    seeds = json.load(open(os.path.join(SD, "noise_seeds.json")))
    for g in seeds:
        v = np.array(g["v"]); dev = v - v.mean()
        x0 = g["k"] - 1 + (-0.18 if g["arm"] == "tgtA" else 0.18) + (-0.06 if g["fold"] == "val_fold0" else 0.06)
        dx.scatter([x0] * 3, dev, s=6, color=ARM_C[g["arm"]], marker="o" if g["fold"] == "val_fold0" else "^",
                   edgecolor="white", linewidth=0.3, zorder=4)
    per = noise["fixed_0_5"]["per_target_two_sigma"]
    for k in (1, 2, 3, 4):
        s2 = per[f"V>={k}"]
        dx.plot([k - 1.38, k - 0.62], [s2, s2], color=INK, lw=0.7); dx.plot([k - 1.38, k - 0.62], [-s2, -s2], color=INK, lw=0.7)
    dx.axhspan(-nf, nf, color=BAND, zorder=0, lw=0)
    dx.axhline(0, color=INK2, lw=0.5)
    dx.set_xticks(range(4), [f"G$_{k}$" for k in (1, 2, 3, 4)])
    dx.set_ylim(-1.6, 1.6)
    dx.set_ylabel("Seed deviation (points)")
    dx.set_title("Retraining noise (3 seeds)", fontsize=6.0)
    dx.legend(handles=[Line2D([], [], marker="o", ls="none", color=BLUE, ms=3, label="arm A"),
                       Line2D([], [], marker="o", ls="none", color=ORANGE, ms=3, label="arm B"),
                       Line2D([], [], color=INK, lw=0.7, label="±2σ per ref.")],
              loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=3, fontsize=5.2, handlelength=1.0, columnspacing=0.5, handletextpad=0.3)
    grid(dx)
    tag(dx, "d", x=-0.45, y=1.04)
    VALUES["fig4d_noise"] = {"pooled_2sigma": nf, "per_target": per}
    save(fig, "Fig4")



# ====================================================================== Fig. 5
from scipy.stats import spearmanr


def _per_nodule_5fold(d, arm, offs):
    fr = []
    for f in range(5):
        pn = pd.DataFrame(d["runs"][f"{arm}_fold{f}"]["per_nodule"]); fr.append(pn)
    x = pd.concat(fr)
    return x.groupby(["case_id", "slice"]).agg({**{f"d{o}": "mean" for o in offs}, "diameter_mm": "first"})


def fig5():
    d5 = J("outputs", "analysis_scope", "diag_offset.json")
    da = J("outputs", "analysis_scope", "diag_offset_allarms.json")
    P = J("outputs", "analysis_scope", "offset_paired_stats.json")
    ex = np.load(os.path.join(SD, "example_slice.npz"))
    offs = [int(o) for o in d5["offsets"]]
    B = _per_nodule_5fold(d5, "tgtB", offs)
    S = _per_nodule_5fold(d5, "tgtBshift", offs)
    assert len(B) == 76

    fig = plt.figure(figsize=(W, 4.6))
    # ---------------- a: what the network sees as the window is displaced
    gsa = fig.add_gridspec(1, 5, left=0.0, right=0.60, top=0.965, bottom=0.735, wspace=0.06)
    img, vote = ex["img"].astype(float), ex["vote"]
    cy, cx = int(round(float(ex["cy"]))), int(round(float(ex["cx"])))
    for j, dd in enumerate([0, 8, 16, 32, 56]):
        ax = fig.add_subplot(gsa[0, j])
        y0, x0 = cy - 64, cx - 64 + dd            # window shifted right: lesion appears left of centre
        crop = img[y0:y0 + 128, x0:x0 + 128]; v = vote[y0:y0 + 128, x0:x0 + 128]
        ax.imshow(crop, cmap="gray", vmin=0, vmax=255, interpolation="bilinear")
        ax.contour((v >= 2).astype(float), [0.5], colors=[ORANGE], linewidths=0.8)
        ax.plot([64], [64], "+", color="white", ms=5, mew=0.7)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color(INK3); sp.set_linewidth(0.5)
        ax.set_title(f"d = {dd} px", fontsize=6.3, pad=2)
        if j == 0:
            tag(ax, "a", x=-0.05, y=1.12)
    # ---------------- b: Dice vs offset
    bx = fig.add_axes([0.075, 0.10, 0.50, 0.555])
    bx.axvspan(0, 32, color=BAND, zorder=0, lw=0)
    others = [r for r in da["runs"] if r not in ("tgtB_fold0", "tgtBshift_fold0")]
    lab_others = {}
    for r in others:
        yv = [da["runs"][r]["mean_by_offset"][str(o)] for o in offs]
        bx.plot(offs, yv, color=GREY, lw=0.7, zorder=2)
        lab_others[r] = yv
    for df_, col, mk, lab in ((B, ORANGE, "o", "B majority, centre-cropped"), (S, INK, "s", "B + translation augmentation (±32 px)")):
        m = [df_[f"d{o}"].mean() for o in offs]
        ci = np.array([boot_mean_ci(df_[f"d{o}"].values, B=3000) for o in offs])
        bx.fill_between(offs, ci[:, 0], ci[:, 1], color=col, alpha=0.16, lw=0, zorder=3)
        bx.plot(offs, m, color=col, lw=1.6, marker=mk, ms=3.4, mec="white", mew=0.5, zorder=5, label=lab)
    bx.plot([], [], color=GREY, lw=0.7, label="nine further arms without translation (fold 0)")
    for r, short in (("archPU1e3_fold0", "Plain U-Net"), ("tgtF_fold0", "SoftSeg"), ("tgtC_fold0", "C majority+soft")):
        yv = lab_others[r]
        ytxt = {"Plain U-Net": 0.875, "SoftSeg": 0.70, "C majority+soft": yv[1] - 0.07}[short]
        bx.annotate(short, xy=(8, yv[1]), xytext=(12.5, ytxt),
                    fontsize=5.3, color=INK2, arrowprops=dict(arrowstyle="-", lw=0.4, color=INK3))
    bx.text(16, 0.955, "augmentation range", ha="center", va="top", fontsize=5.6, color=INK2)
    bx.set_xticks(offs)
    bx.set_xlim(-1.5, 57.5); bx.set_ylim(0, 0.97)
    bx.set_xlabel("Offset of the evaluation window from the lesion centroid, d (px)")
    bx.set_ylabel("Dice against G$_2$ (held-out test, 76 nodules)")
    bx.legend(loc="lower left", bbox_to_anchor=(0.0, 0.0), fontsize=5.5, handlelength=1.6)
    grid(bx)
    tag(bx, "b", x=-0.13, y=1.0)
    VALUES["fig5b"] = {"B_5fold": [round(B[f"d{o}"].mean(), 4) for o in offs],
                       "Bshift_5fold": [round(S[f"d{o}"].mean(), 4) for o in offs]}

    # ---------------- c: heat strip, all eleven arms (fold 0)
    cxax = fig.add_axes([0.70, 0.535, 0.21, 0.43])
    runs = da["runs"]
    names = {"tgtA_fold0": "A union", "tgtB_fold0": "B majority", "tgtC_fold0": "C maj.+soft", "tgtD_fold0": "D soft only",
             "tgtE_fold0": "E SVLS", "tgtF_fold0": "F SoftSeg", "tgtBshift_fold0": "B + translation",
             "archR18_fold0": "ResNet-18", "archCNX_fold0": "ConvNeXt-T", "archPVT_fold0": "PVT v2-b1",
             "archPU1e3_fold0": "Plain U-Net"}
    order = sorted(runs, key=lambda r: runs[r]["mean_by_offset"]["0"] - runs[r]["mean_by_offset"]["8"])
    M = np.array([[runs[r]["mean_by_offset"][str(o)] for o in offs] for r in order])
    cmap = mpl.colors.LinearSegmentedColormap.from_list("b", ["#f4f8fd", "#9ec5f4", "#2a78d6", "#0d366b"])
    im = cxax.imshow(M, cmap=cmap, vmin=0, vmax=0.9, aspect="auto")
    cxax.set_yticks(range(len(order)), [names[r] for r in order], fontsize=5.4)
    cxax.set_xticks(range(len(offs)), offs, fontsize=5.2)
    cxax.set_xlabel("d (px)", fontsize=5.8)
    cxax.tick_params(length=0)
    for sp in cxax.spines.values():
        sp.set_visible(False)
    for i, r in enumerate(order):
        drop = 100 * (runs[r]["mean_by_offset"]["0"] - runs[r]["mean_by_offset"]["8"])
        cxax.text(len(offs) - 0.3, i, sgn(-drop, 1), va="center", ha="left", fontsize=5.2,
                  color=INK if "shift" not in r else ORANGE)
    cxax.text(len(offs) - 0.3, -0.9, "Δ at d=8", va="center", ha="left", fontsize=5.2, color=INK2)
    cb = fig.colorbar(im, ax=cxax, fraction=0.06, pad=0.20, aspect=16, location="right")
    cb.ax.tick_params(labelsize=5, length=1.5, colors=INK2); cb.outline.set_visible(False)
    cb.set_label("Dice", fontsize=5.5, color=INK2)
    tag(cxax, "c", x=-0.75, y=1.01)
    VALUES["fig5c_drop8"] = {names[r]: round(100 * (runs[r]["mean_by_offset"]["0"] - runs[r]["mean_by_offset"]["8"]), 2) for r in order}
    VALUES["fig5c_median_drop8_10arms"] = P["ten_arms_fold0"]["median_drop8_pts"]

    # ---------------- d: size dependence of the position prior
    dx = fig.add_axes([0.70, 0.10, 0.29, 0.30])
    dropB = 100 * (B.d0 - B.d8); dropS = 100 * (S.d0 - S.d8)
    dx.scatter(B.diameter_mm, dropB, s=7, color=ORANGE, edgecolor="white", lw=0.3, zorder=4, label="B majority")
    dx.scatter(S.diameter_mm, dropS, s=7, facecolor="white", edgecolor=INK, lw=0.5, marker="s", zorder=3, label="B + translation")
    dx.axhline(0, color=INK2, lw=0.5)
    rB = spearmanr(B.diameter_mm, dropB); rS = spearmanr(S.diameter_mm, dropS)
    def sci(p):
        m, e = f"{p:.1e}".split("e")
        return f"${m}\\times10^{{{int(e)}}}$"
    dx.text(0.98, 0.97, f"B: ρ = {num(rB.correlation)}, p = {sci(rB.pvalue)}\nB + translation: ρ = {num(rS.correlation)}, p = {rS.pvalue:.2f}",
            transform=dx.transAxes, ha="right", va="top", fontsize=5.1, color=INK2)
    dx.set_xscale("log"); dx.set_xticks([3, 5, 10, 20], ["3", "5", "10", "20"]); dx.minorticks_off()
    dx.set_xlabel("Nodule equivalent diameter (mm)")
    dx.set_ylabel("Dice lost at d = 8 (points)")
    dx.legend(loc="center right", bbox_to_anchor=(1.0, 0.52), fontsize=5.1, handletextpad=0.2)
    grid(dx)
    tag(dx, "d", x=-0.30, y=1.02)
    bins = [3, 6, 10, 30]
    VALUES["fig5d"] = {"spearman_B": [round(rB.correlation, 3), float(rB.pvalue)],
                       "spearman_shift": [round(rS.correlation, 3), float(rS.pvalue)],
                       "mean_drop_by_diam": {str(k): round(v, 2) for k, v in dropB.groupby(pd.cut(B.diameter_mm, bins)).mean().items()},
                       "n_by_diam": {str(k): int(v) for k, v in pd.cut(B.diameter_mm, bins).value_counts().sort_index().items()}}
    save(fig, "Fig5")



# ====================================================================== Fig. 6


SCOPES = ["patch128", "win128", "win192", "win256", "win384", "win512", "slice", "volume"]
SC_L = ["patch\n128", "win\n128", "win\n192", "win\n256", "win\n384", "win\n512", "full\nslice", "whole\nvolume"]
ARMS = [("tgtB", "B majority (centre-cropped)", ORANGE, "o"), ("bg_maj", "B + background sampling", AQUA, "^"),
        ("tgtBshift", "B + translation augmentation", VIOLET, "s"), ("tgtBshift_bg", "B + both remedies", BLUE, "D")]


def fig6():
    S = J("outputs", "analysis_scope", "scope_analysis_merged2.json")
    T = thr_cv()
    sc = scope_case()
    ex = np.load(os.path.join(SD, "example_slice.npz"))
    fig = plt.figure(figsize=(W, 4.35))

    # ---------------- a: nested evaluation windows on a real slice
    ax = fig.add_axes([0.0, 0.50, 0.30, 0.47])
    img = ex["img"].astype(float); vote = ex["vote"]
    cy, cx = float(ex["cy"]), float(ex["cx"])
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="bilinear")
    ax.contour((vote >= 2).astype(float), [0.5], colors=[ORANGE], linewidths=0.6)
    ramp = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5"]
    for w, c in zip((128, 192, 256, 384, 512), ramp):
        x0 = np.clip(cx - w / 2, 0, 512 - w); y0 = np.clip(cy - w / 2, 0, 512 - w)
        ax.add_patch(mpl.patches.Rectangle((x0, y0), w - 1, w - 1, fill=False, ec=c, lw=0.9 if w < 512 else 1.6))
        if w < 512:
            ax.text(x0 + 5, y0 + 5, f"{w}", color="white", fontsize=5.0, va="top", ha="left", path_effects=halo(1.4, "#000000"))
    ax.set_xlim(0, 511); ax.set_ylim(511, 0)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("One forward pass on the full slice;\nonly the scoring window changes", fontsize=6.0)
    tag(ax, "a", x=0.0, y=1.0)

    # ---------------- b: ladder with 95% CI
    bx = fig.add_axes([0.385, 0.565, 0.60, 0.385])
    x = np.arange(len(SCOPES))
    bx.axvspan(0.5, 5.5, color=BAND, zorder=0, lw=0)
    bx.text(3.0, 0.985, "field-of-view ladder (one probability map, five windows)", ha="center", va="top", fontsize=5.6, color=INK2)
    lad = {}
    for a, lab, col, mk in ARMS:
        t = T[a]["2"]["thr"]
        s = sc[(sc.arm == a) & np.isclose(sc.threshold, t)]
        m, lo, hi = [], [], []
        for scp in SCOPES:
            v = s[s.scope == scp].dice.values
            m.append(v.mean()); l_, h_ = boot_mean_ci(v, B=2000); lo.append(l_); hi.append(h_)
        assert abs(m[1] - S["table1_ladder"][f"{a}_v2"]["win128"]) < 1e-3, (a, m[1])
        bx.fill_between(x, lo, hi, color=col, alpha=0.18, lw=0, zorder=2)
        bx.plot(x, m, color=col, lw=1.3, marker=mk, ms=3.2, mec="white", mew=0.5, zorder=4, label=lab)
        lad[a] = [round(v, 4) for v in m]
    bx.set_xticks(x, SC_L, fontsize=5.6)
    bx.set_xlim(-0.4, len(SCOPES) - 0.6); bx.set_ylim(0, 1.0)
    bx.set_ylabel("Dice against G$_2$\n(held-out test, n = 158)")
    bx.legend(loc="upper right", bbox_to_anchor=(1.0, 0.90), fontsize=5.4, handlelength=1.6)
    grid(bx)
    tag(bx, "b", x=-0.10, y=1.0)
    VALUES["fig6b_ladder"] = lad

    # ---------------- c: per-case ladder loss
    cxx = fig.add_axes([0.07, 0.09, 0.40, 0.30])
    rng = np.random.default_rng(5)
    loss_out = {}
    for i, (a, lab, col, mk) in enumerate(ARMS):
        t = T[a]["2"]["thr"]
        s = sc[(sc.arm == a) & np.isclose(sc.threshold, t)].pivot(index="case_id", columns="scope", values="dice")
        d = 100 * (s["win128"] - s["win512"]).dropna().values
        cxx.scatter(rng.uniform(-0.2, 0.2, len(d)) + i, d, s=2.0, color=col, alpha=0.5, lw=0, zorder=3, rasterized=True)
        q1, med, q3 = np.percentile(d, [25, 50, 75])
        cxx.plot([i + 0.27, i + 0.27], [q1, q3], color=INK, lw=2.0, solid_capstyle="butt", zorder=4)
        cxx.plot([i + 0.22, i + 0.32], [med, med], color=INK, lw=0.8, zorder=5)
        r = [q for q in S["table4_paired"] if q["arm"] == a and q["level"] == 2 and q["segment"] == "纯视野效应"][0]
        assert abs(-r["diff"] - d.mean()) < 0.01, (a, r["diff"], d.mean())
        lo, hi = -r["ci"][1], -r["ci"][0]          # the paired CI reported in the text (analyze_scope.py)
        cxx.text(i, 96, f"{num(d.mean())}\n[{num(lo)}, {num(hi)}]", ha="center", va="top", fontsize=5.0, color=INK2, linespacing=1.1)
        loss_out[a] = {"n": int(len(d)), "mean": round(d.mean(), 2), "ci": [round(lo, 2), round(hi, 2)], "median": round(med, 2)}
    cxx.axhline(0, color=INK2, lw=0.5)
    cxx.set_xticks(range(4), ["B", "+ background", "+ translation", "+ both"], fontsize=5.8)
    cxx.set_ylim(-35, 100)
    cxx.set_ylabel("Dice lost, win 128 → 512\nper case (points)")
    grid(cxx)
    tag(cxx, "c", x=-0.20, y=1.02)
    VALUES["fig6c_loss"] = loss_out

    # ---------------- d: threshold drift and recalibration
    dx = fig.add_axes([0.585, 0.09, 0.40, 0.30])
    t3 = S["table3_per_scope_opt"]
    lad_sc = SCOPES[1:6]
    xx = np.arange(len(lad_sc))
    drift = {}
    for a, lab, col, mk in ARMS[2:]:
        fixed = [S["table1_ladder"][f"{a}_v2"][s_] for s_ in lad_sc]
        opt = [t3[f"{a}_v2_{s_}"]["dice_opt"] for s_ in lad_sc]
        thr = [t3[f"{a}_v2_{s_}"]["thr"] for s_ in lad_sc]
        dx.plot(xx, fixed, color=col, lw=1.2, marker=mk, ms=3.0, mec="white", mew=0.4, zorder=4)
        dx.plot(xx, opt, color=col, lw=1.0, ls=(0, (3, 1.5)), marker=mk, ms=3.0, mfc="white", mec=col, mew=0.6, zorder=4)
        up = a == "tgtBshift_bg"
        for xi, (o, t) in enumerate(zip(opt, thr)):
            dx.text(xi + (0.0 if up else 0.16), o + (0.022 if up else -0.018), f"{t:g}".replace("0.", "."), ha="center",
                    va="bottom" if up else "top", fontsize=4.8, color=col)
        drift[a] = {"thr_per_scope": thr, "loss_fixed_pts": round(100 * (fixed[0] - fixed[-1]), 2),
                    "loss_recalibrated_pts": round(100 * (opt[0] - opt[-1]), 2)}
    from matplotlib.lines import Line2D
    dx.legend(handles=[Line2D([], [], color=INK2, lw=1.2, label="validation-calibrated threshold"),
                       Line2D([], [], color=INK2, lw=1.0, ls=(0, (3, 1.5)), label="re-optimised per window (labels: threshold)")],
              loc="lower left", fontsize=5.1, handlelength=1.8)
    dx.set_xticks(xx, ["128", "192", "256", "384", "512"])
    dx.set_xlabel("Evaluation window (px)")
    dx.set_ylabel("Dice against G$_2$")
    dx.set_ylim(0.2, 0.86)
    grid(dx)
    tag(dx, "d", x=-0.17, y=1.02)
    dx.text(0.98, 0.97, "violet: + translation   blue: + both", transform=dx.transAxes, ha="right", va="top", fontsize=5.0, color=INK2)
    VALUES["fig6d_drift"] = drift
    save(fig, "Fig6")



# ====================================================================== Fig. 7


def effect_rows():
    """Ranked effect sizes. Test-set values and CIs come from effect_uncertainty.json / scope / offset JSON
    (the numbers quoted in the text); CV values from symmetric_analysis.json."""
    E = J("outputs", "analysis", "effect_uncertainty.json")["own_sample_bootstrap"]
    O = J("outputs", "analysis_scope", "offset_paired_stats.json")["ten_arms_fold0"]
    N = J("outputs", "analysis_full", "noise_floor.json")
    T = thr_cv()
    cv_tgt = 100 * (max(T["tgtA"][k]["dice_05"] for k in "1234") - min(T["tgtA"][k]["dice_05"] for k in "1234"))
    cv_thr = 100 * (T["tgtA"]["4"]["dice_opt"] - T["tgtA"]["4"]["dice_05"])
    cv_sup = 100 * (max(T[a]["4"]["dice_opt"] for a in ("tgtA", "tgtB", "tgtC", "tgtD", "tgtE")) -
                    min(T[a]["4"]["dice_opt"] for a in ("tgtA", "tgtB", "tgtC", "tgtD", "tgtE")))
    cv_arch = 100 * (max(T[a]["2"]["dice_opt"] for a in ("tgtB", "archR18", "archCNX", "archPVT", "archPU1e3")) -
                     min(T[a]["2"]["dice_opt"] for a in ("tgtB", "archR18", "archCNX", "archPVT", "archPU1e3")))
    cv_shift = 100 * (T["tgtB"]["2"]["dice_opt"] - T["tgtBshift"]["2"]["dice_opt"])
    rows = [
        ("Field of view 128→512 px, position-invariant arm", -E["fov_position_invariant_pts"]["point"],
         sorted(-np.array(E["fov_position_invariant_pts"]["ci95"])), None, "protocol", None),
        ("Evaluation target V≥1 vs V≥4, arm A, threshold 0.5", E["evaluation_target_span_pts"]["point"],
         E["evaluation_target_span_pts"]["ci95"], cv_tgt, "protocol", None),
        ("Lesion offset 8 px (median of 10 arms)", O["median_drop8_pts"], None, None, "data",
         sorted(v["drop8_pts"] for v in O["arms"].values())),
        ("Field of view 128→512 px, both confounds removed", -E["fov_effect_pts"]["point"],
         sorted(-np.array(E["fov_effect_pts"]["ci95"])), None, "protocol", None),
        ("Binarisation threshold, arm A at V≥4", E["threshold_effect_pts"]["point"], E["threshold_effect_pts"]["ci95"],
         cv_thr, "protocol", None),
        ("Cost of translation augmentation (centred test)", E["translation_cost_pts"]["point"],
         E["translation_cost_pts"]["ci95"], cv_shift, "data", None),
        ("Supervision target, 5 arms at V≥4", E["supervision_range_pts"]["point"], E["supervision_range_pts"]["ci95"],
         cv_sup, "method", None),
        ("Architecture, 5 models at V≥2", E["architecture_range_pts"]["point"], E["architecture_range_pts"]["ci95"],
         cv_arch, "method", None),
    ]
    rows.sort(key=lambda r: -r[1])
    nf = N["fixed_0_5"]["two_sigma_pts"]
    rows.append(("Retraining with a new seed (2σ, validation)", nf, [0.96, 1.58], nf, "noise", None))
    return rows, nf


def fig7():
    rows, nf = effect_rows()
    fig = plt.figure(figsize=(W, 4.3))
    ax = fig.add_axes([0.335, 0.375, 0.40, 0.615])
    y = np.arange(len(rows))[::-1]
    ax.axvspan(0, nf, color=BAND, zorder=0, lw=0)
    out = []
    for yi, (lab, v, ci, cvv, cat, dots) in zip(y, rows):
        col = CAT_C[cat]
        ax.barh(yi, v, height=0.56, color=col, alpha=0.88, edgecolor="white", lw=0.6, zorder=3)
        right = v
        if ci is not None:
            ax.plot(ci, [yi, yi], color=INK, lw=0.7, zorder=5)
            for b in ci:
                ax.plot([b, b], [yi - 0.11, yi + 0.11], color=INK, lw=0.7, zorder=5)
            right = max(right, ci[1])
        if dots:
            ax.plot(dots, [yi] * len(dots), "o", ms=2.4, color="white", mec=INK, mew=0.5, zorder=6)
            right = max(right, max(dots))
        if cvv is not None and cat != "noise":
            ax.plot([cvv], [yi], "D", ms=3.0, color="white", mec=INK, mew=0.7, zorder=7)
        ax.text(right + 0.8, yi, num(v), va="center", ha="left", fontsize=6.0, fontweight="bold", zorder=6)
        out.append({"row": lab, "test": round(v, 2), "ci": None if ci is None else [round(c, 2) for c in ci],
                    "cv": None if cvv is None else round(cvv, 2)})
    ax.set_yticks(y, [r[0] for r in rows], fontsize=5.9)
    ax.tick_params(axis="y", length=0, colors=INK)
    ax.set_xlim(0, 50)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.spines["left"].set_visible(False)
    grid(ax, "x")
    ax.set_xlabel("Change in Dice (points), one factor varied\n(bars: held-out test; open diamonds: cross-validation)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=CAT_C["protocol"], label="evaluation protocol"),
                       Patch(color=CAT_C["data"], label="data / training protocol"),
                       Patch(color=CAT_C["method"], label="learning design"),
                       Patch(color=CAT_C["noise"], label="run-to-run noise")],
              loc="lower right", fontsize=5.4, handlelength=1.0, handleheight=0.8)
    tag(ax, "a", x=-0.85, y=0.985)
    VALUES["fig7a"] = out

    # ---------------- b: in multiples of the noise floor
    bx = fig.add_axes([0.775, 0.375, 0.215, 0.615])
    for yi, (lab, v, ci, cvv, cat, dots) in zip(y, rows):
        if cat == "noise":
            continue
        col = CAT_C[cat]
        bx.plot([v / nf], [yi], "o", ms=3.4, color=col, zorder=5)
        if ci is not None:
            bx.plot([max(ci[0], 0.05) / nf, ci[1] / nf], [yi, yi], color=col, lw=1.0, zorder=4)
    bx.axvline(1, color=INK2, lw=0.6, ls=(0, (3, 2)))
    bx.axvline(10, color=INK3, lw=0.5, ls=(0, (1, 2)))
    bx.set_xscale("log")
    bx.set_xlim(0.3, 60)
    bx.set_xticks([0.5, 1, 2, 5, 10, 20, 50], ["0.5", "1", "2", "5", "10", "20", "50"])
    bx.minorticks_off()
    bx.set_yticks(y[:-1], [])
    bx.set_ylim(-0.6, len(rows) - 0.4)
    bx.set_xlabel("Effect ÷ noise floor (×)")
    grid(bx, "x")
    tag(bx, "b", x=-0.10, y=0.985)

    # ---------------- c: joint bootstrap of the two headline ratios (124 common cases)
    Tt = J("outputs", "analysis_test", "symmetric_analysis_applied.json")["table1"]
    T = thr_cv()
    sup = pd.concat({a: pc_test(a, 4, Tt[a]["4"]["thr"]) for a in ("tgtA", "tgtB", "tgtC", "tgtD", "tgtE")}, axis=1)
    arch = pd.concat({a: pc_test(a, 2, Tt[a]["2"]["thr"]) for a in ("tgtB", "archR18", "archCNX", "archPVT", "archPU1e3")}, axis=1)
    arch.columns = [f"arch_{c}" for c in arch.columns]
    sc = scope_case()
    t = T["tgtBshift_bg"]["2"]["thr"]
    s = sc[(sc.arm == "tgtBshift_bg") & np.isclose(sc.threshold, t)].pivot(index="case_id", columns="scope", values="dice")
    fov = (s["win128"] - s["win512"]).rename("fov")
    base = sup.join(arch, how="inner").join(fov, how="inner").dropna()
    rng = np.random.default_rng(42)
    idx = np.arange(len(base))
    r1, r2 = [], []
    S_ = base[["tgtA", "tgtB", "tgtC", "tgtD", "tgtE"]].values
    A_ = base[[c for c in base.columns if c.startswith("arch_")]].values
    F_ = base["fov"].values
    for _ in range(5000):
        b = rng.choice(idx, len(idx), replace=True)
        fv = F_[b].mean(); sm = S_[b].mean(0); am = A_[b].mean(0)
        r1.append(fv / (sm.max() - sm.min())); r2.append(fv / (am.max() - am.min()))
    p1 = F_.mean() / (S_.mean(0).max() - S_.mean(0).min()); p2 = F_.mean() / (A_.mean(0).max() - A_.mean(0).min())
    cx = fig.add_axes([0.335, 0.075, 0.655, 0.14])
    binsl = np.logspace(np.log10(1), np.log10(80), 60)
    for r, col, lab, pt in ((r1, ORANGE, "÷ supervision range", p1), (r2, INK2, "÷ architecture range", p2)):
        cx.hist(r, bins=binsl, color=col, alpha=0.55, lw=0, zorder=3)
        lo, hi = np.percentile(r, [2.5, 97.5])
        cx.plot([lo, hi], [cx.get_ylim()[1] * 0 + 0, 0], color=col, lw=0)
        VALUES[f"fig7c_{'sup' if col == ORANGE else 'arch'}"] = {"point": round(pt, 2), "ci95": [round(lo, 2), round(hi, 2)],
                                                                   "n_common": int(len(base))}
    ymax = cx.get_ylim()[1]
    JR = J("outputs", "analysis", "effect_uncertainty.json")["joint_ratio_bootstrap"]
    for r, col, pt, lab, key in ((r1, ORANGE, p1, "FOV ÷ supervision", "ratio_protocol_supervision"),
                                 (r2, INK2, p2, "FOV ÷ architecture", "ratio_protocol_architecture")):
        pt = JR[key]["point"]; lo, hi = JR[key]["ci95"]
        yy = ymax * (1.05 if col == ORANGE else 1.22)
        cx.plot([lo, hi], [yy, yy], color=col, lw=1.0, clip_on=False)
        cx.plot([pt], [yy], "o", ms=2.8, color=col, clip_on=False)
        cx.text(hi * 1.12, yy, f"{lab}: {pt:.2f}× [{lo:.2f}, {hi:.2f}]", fontsize=5.3, va="center", color=INK2, clip_on=False)
    cx.axvline(1, color=INK2, lw=0.6, ls=(0, (3, 2)))
    cx.set_xscale("log"); cx.set_xlim(0.8, 80)
    cx.set_xticks([1, 2, 5, 10, 20, 50], ["1", "2", "5", "10", "20", "50"]); cx.minorticks_off()
    cx.set_yticks([])
    cx.spines["left"].set_visible(False)
    cx.set_xlabel("Joint bootstrap ratio (×), 124 cases")
    tag(cx, "c", x=-0.52, y=0.9)
    fig.text(0.03, 0.155, "Conservative field-of-view effect divided by\nthe two learning-design ranges: joint case-level\nbootstrap, 5000 resamples of the 124 test cases\nwith a non-empty V ≥ 4 reference", fontsize=5.5,
             color=INK2, va="center", ha="left", linespacing=1.25)
    assert abs(p1 - 7.03) < 0.05 and abs(p2 - 15.57) < 0.1, (p1, p2)
    save(fig, "Fig7")



# ====================================================================== Fig. 8


def _span_boot(frame, B=3000, seed=0):
    rng = np.random.default_rng(seed)
    v = frame.values
    pt = np.nanmean(v, 0); pt = 100 * (np.nanmax(pt) - np.nanmin(pt))
    bs = []
    for _ in range(B):
        m = np.nanmean(v[rng.integers(0, len(v), len(v))], 0)
        bs.append(100 * (np.nanmax(m) - np.nanmin(m)))
    return pt, np.percentile(bs, [2.5, 97.5])


def fig8():
    Tt = J("outputs", "analysis_test", "symmetric_analysis_applied.json")["table1"]
    T = thr_cv()
    CL = J("outputs", "analysis", "case_lesion_sensitivity.json")["sets"]
    LU = json.load(open(os.path.join(SD, "lesion_unit.json")))
    fig = plt.figure(figsize=(W, 4.0))

    # ---------------- a: nodule-size strata (held-out test)
    ax = fig.add_axes([0.075, 0.56, 0.52, 0.40])
    strata = [("all", "all nodules"), ("3-5", "3–5 mm"), ("5-10", "5–10 mm"), ("10-30", "10–30 mm")]
    effects = [("Evaluation target", BLUE, "o"), ("Threshold", BLUE, "s"), ("Supervision target", ORANGE, "o"),
               ("Architecture", ORANGE, "s")]
    res = {}
    for si, (st, stl) in enumerate(strata):
        ev = pd.concat({k: pc_test("tgtA", k, 0.5, st) for k in (1, 2, 3, 4)}, axis=1)
        a1 = pc_test("tgtA", 4, Tt["tgtA"]["4"]["thr"], st); a0 = pc_test("tgtA", 4, 0.5, st)
        a1, a0 = a1.align(a0, join="inner"); td = 100 * (a1 - a0).values
        sup = pd.concat({a: pc_test(a, 4, Tt[a]["4"]["thr"], st) for a in ("tgtA", "tgtB", "tgtC", "tgtD", "tgtE")}, axis=1).dropna()
        arc = pd.concat({a: pc_test(a, 2, Tt[a]["2"]["thr"], st) for a in ("tgtB", "archR18", "archCNX", "archPVT", "archPU1e3")}, axis=1).dropna()
        vals = [_span_boot(ev, seed=1), (td.mean(), boot_mean_ci(td, B=3000)), _span_boot(sup, seed=2), _span_boot(arc, seed=3)]
        ns = [ev.shape[0], len(td), sup.shape[0], arc.shape[0]]
        res[st] = {e[0]: {"pt": round(v[0], 2), "ci": [round(c, 2) for c in v[1]], "n": n} for e, v, n in zip(effects, vals, ns)}
        for ei, ((name, col, mk), (pt, ci)) in enumerate(zip(effects, vals)):
            x = si + (ei - 1.5) * 0.18
            ax.plot([x, x], ci, color=col, lw=0.9, zorder=3)
            ax.plot([x], [pt], mk, ms=3.6, color=col if mk == "o" else "white", mec=col, mew=0.8, zorder=4)
        strata[si] = (st, f"{stl}\n({ns[0]} cases)")
    ax.set_xticks(range(4), [s[1] for s in strata], fontsize=5.8)
    ax.set_xlim(-0.5, 3.5); ax.set_ylim(0, 52)
    ax.axhspan(0, 1.19, color=BAND, zorder=0, lw=0)
    ax.set_ylabel("Effect size (Dice points)")
    ax.set_title("Held-out test, by nodule size (case level within stratum)", fontsize=6.4)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker=m, ls="none", color=c, mfc=c if m == "o" else "white", mec=c, ms=3.6, label=n)
                       for n, c, m in effects], loc="upper right", fontsize=5.5, ncol=2, columnspacing=0.8, handletextpad=0.2)
    grid(ax)
    tag(ax, "a", x=-0.12, y=1.02)
    VALUES["fig8a_strata"] = res

    # ---------------- b: unit of analysis
    bx = fig.add_axes([0.68, 0.56, 0.31, 0.40])
    names = ["Evaluation\ntarget", "Threshold", "Supervision", "Architecture"]
    case_cv = [25.58, 9.42, 2.65, 0.96]
    case_te = [28.58, 9.99, 3.12, 1.19]
    les_cv = [CL["cv"]["protocol_effect"]["lesion"]["range_pts"], LU["cv_tgtA_v4_lesion_unit"]["threshold_effect_pts"],
              CL["cv"]["supervision_range"]["lesion"]["range_pts"], CL["cv"]["architecture_range"]["lesion"]["range_pts"]]
    les_te = [CL["test"]["protocol_effect"]["lesion"]["range_pts"], LU["test_tgtA_v4_lesion_unit"]["threshold_effect_pts"],
              CL["test"]["supervision_range"]["lesion"]["range_pts"], CL["test"]["architecture_range"]["lesion"]["range_pts"]]
    cols = [BLUE, BLUE, ORANGE, ORANGE]
    for i in range(4):
        for dx_, c_, l_, mk in ((-0.12, case_cv[i], les_cv[i], "o"), (0.12, case_te[i], les_te[i], "D")):
            bx.plot([i + dx_ - 0.05, i + dx_ + 0.05], [c_, l_], color=cols[i], lw=0.8, zorder=3)
            bx.plot([i + dx_ - 0.05], [c_], mk, ms=3.2, color=cols[i], zorder=4)
            bx.plot([i + dx_ + 0.05], [l_], mk, ms=3.2, color="white", mec=cols[i], mew=0.8, zorder=4)
    bx.set_yscale("log"); bx.set_ylim(0.6, 45)
    bx.set_yticks([1, 2, 5, 10, 20, 40], ["1", "2", "5", "10", "20", "40"]); bx.minorticks_off()
    bx.set_xticks(range(4), names, fontsize=5.6)
    bx.set_ylabel("Effect size (Dice points, log)")
    bx.set_title("Unit of analysis: case (filled) vs lesion (open)", fontsize=6.2)
    bx.text(0.02, 0.03, "circles: CV   diamonds: test", transform=bx.transAxes, fontsize=5.2, color=INK2)
    grid(bx)
    tag(bx, "b", x=-0.22, y=1.02)
    VALUES["fig8b_units"] = {"case_cv": case_cv, "lesion_cv": [round(v, 2) for v in les_cv],
                             "case_test": case_te, "lesion_test": [round(v, 2) for v in les_te]}

    # ---------------- c: cross-evaluation matrix, CV vs test
    cx = fig.add_axes([0.075, 0.10, 0.36, 0.33])
    xs, ys = [], []
    for a in ("tgtA", "tgtB", "tgtC", "tgtD", "tgtE", "archR18", "archCNX", "archPVT", "archPU1e3"):
        for k in "1234":
            for key, mk in (("dice_05", "o"), ("dice_opt", "D")):
                x_, y_ = 100 * T[a][k][key], 100 * Tt[a][k][key]
                xs.append(x_); ys.append(y_)
                cx.plot([x_], [y_], mk, ms=2.4, color=REF_C[int(k) - 1], mec="white" if mk == "o" else REF_C[int(k) - 1],
                        mfc=REF_C[int(k) - 1] if mk == "o" else "white", mew=0.5, zorder=3)
    cx.plot([55, 90], [55, 90], color=INK3, lw=0.6, ls=(0, (3, 2)))
    r = np.corrcoef(xs, ys)[0, 1]; mad = np.mean(np.abs(np.array(xs) - np.array(ys)))
    cx.text(0.04, 0.95, f"72 cells, r = {r:.3f}\nmean |Δ| = {mad:.2f} points", transform=cx.transAxes, va="top", fontsize=5.4, color=INK2)
    cx.text(0.97, 0.04, "colour: reference G$_1$ → G$_4$\nfilled: threshold 0.5; open: calibrated", transform=cx.transAxes,
            va="bottom", ha="right", fontsize=5.0, color=INK2)
    cx.set_xlim(55, 90); cx.set_ylim(55, 90)
    cx.set_xlabel("Cross-validation Dice (points)"); cx.set_ylabel("Held-out test Dice (points)")
    cx.set_title("Cross-evaluation matrix, 9 arms × 4 references × 2 operating points", fontsize=5.9)
    grid(cx, "both")
    tag(cx, "c", x=-0.22, y=1.04)
    VALUES["fig8c"] = {"r": round(r, 4), "mean_abs_diff": round(mad, 3), "n": len(xs)}

    # ---------------- d: paired contrasts, CV vs test
    dx = fig.add_axes([0.58, 0.10, 0.41, 0.33])
    cmap_c = {("tgtC", "tgtA"): BLUE, ("tgtB", "tgtA"): ORANGE, ("tgtC", "tgtB"): VIOLET, ("tgtD", "tgtB"): AQUA,
              ("tgtD", "tgtC"): GREY, ("tgtC", "tgtE"): MAGENTA}
    xs, ys = [], []
    sign_agree = 0
    for (a, b), col in cmap_c.items():
        for k in (1, 2, 3, 4):
            dcv, dte = _paired(a, b, k, False), _paired(a, b, k, True)
            lo1, hi1 = boot_mean_ci(dcv, B=1500); lo2, hi2 = boot_mean_ci(dte, B=1500)
            x_, y_ = dcv.mean(), dte.mean()
            dx.plot([lo1, hi1], [y_, y_], color=col, lw=0.5, alpha=0.7, zorder=2)
            dx.plot([x_, x_], [lo2, hi2], color=col, lw=0.5, alpha=0.7, zorder=2)
            dx.plot([x_], [y_], "o", ms=3.0, color=col, mec="white", mew=0.4, zorder=3)
            xs.append(x_); ys.append(y_); sign_agree += np.sign(x_) == np.sign(y_)
    dx.plot([-4, 2], [-4, 2], color=INK3, lw=0.6, ls=(0, (3, 2)))
    dx.axhline(0, color=INK3, lw=0.4); dx.axvline(0, color=INK3, lw=0.4)
    r = np.corrcoef(xs, ys)[0, 1]
    dx.text(0.04, 0.95, f"24 contrasts, r = {r:.2f}\nsame sign in {sign_agree}/24", transform=dx.transAxes, va="top",
            fontsize=5.4, color=INK2)
    dx.set_xlim(-3.9, 1.9); dx.set_ylim(-3.9, 1.9)
    dx.set_xlabel("Cross-validation contrast (points)"); dx.set_ylabel("Held-out test contrast (points)")
    dx.set_title("Paired contrasts (Fig. 4a), with 95% CIs", fontsize=5.9)
    from matplotlib.lines import Line2D
    labs = {("tgtC", "tgtA"): "C − A", ("tgtB", "tgtA"): "B − A", ("tgtC", "tgtB"): "C − B", ("tgtD", "tgtB"): "D − B",
            ("tgtD", "tgtC"): "D − C", ("tgtC", "tgtE"): "C − E"}
    dx.legend(handles=[Line2D([], [], marker="o", ls="none", color=c, ms=3, label=labs[k]) for k, c in cmap_c.items()],
              loc="lower right", fontsize=5.2, ncol=2, columnspacing=0.6, handletextpad=0.1)
    grid(dx, "both")
    tag(dx, "d", x=-0.17, y=1.04)
    VALUES["fig8d"] = {"r": round(r, 3), "same_sign": int(sign_agree)}
    save(fig, "Fig8")



# ====================================================================== Fig. 9
import itertools
from scipy.stats import rankdata


def _exact(x, y):
    rx = rankdata(x) - rankdata(x).mean(); ry0 = rankdata(y)
    obs = float(rx @ (ry0 - ry0.mean()) / np.sqrt((rx @ rx) * ((ry0 - ry0.mean()) @ (ry0 - ry0.mean()))))
    hit = tot = 0
    for perm in itertools.permutations(range(len(x))):
        ry = ry0[list(perm)] - ry0.mean()
        r = rx @ ry / np.sqrt((rx @ rx) * (ry @ ry))
        tot += 1; hit += abs(r) >= abs(obs) - 1e-12
    return obs, hit / tot


TASKS = [("LIDC", "LIDC-IDRI", 4, "CT"), ("QUBIQ-brain-tumor-task2", "brain-tumour t2", 3, "MRI"),
         ("QUBIQ-brain-growth", "brain-growth", 7, "MRI"), ("QUBIQ-brain-tumor-task3", "brain-tumour t3", 3, "MRI"),
         ("QUBIQ-prostate-task2", "prostate t2", 6, "MRI"), ("QUBIQ-brain-tumor-task1", "brain-tumour t1", 3, "MRI"),
         ("QUBIQ-kidney", "kidney", 3, "CT"), ("QUBIQ-prostate-task1", "prostate t1", 6, "MRI")]


def fig9():
    dis = J("outputs", "analysis", "disagreement_index.json")
    eff = J("outputs", "analysis", "qubiq_all_effects.json")
    gen = J("outputs", "analysis", "generalisation_table.json")
    fam = J("outputs", "analysis_full", "qubiq_family_sensitivity.json")
    curves = json.load(open(os.path.join(SD, "qubiq_curves.json")))
    T = thr_cv()
    dmap = {"QUBIQ-brain-growth": "QUBIQ-brain"}
    pts = {}
    for key, name, n, mod in TASKS:
        if key == "LIDC":
            pts[key] = (gen["agreement_index"]["LIDC"], gen["LIDC"]["opt_thr"]["protocol"], gen["LIDC"]["opt_thr"]["supervision"])
        else:
            o = eff[key]["effects"]["opt_thr"]
            pts[key] = (dis[dmap.get(key, key)]["area_mean_ratio"], o["protocol"], o["supervision"])
    keys = [t[0] for t in TASKS]
    x = np.array([pts[k][0] for k in keys]); pr = np.array([pts[k][1] for k in keys]); su = np.array([pts[k][2] for k in keys])
    q = [i for i, k in enumerate(keys) if k != "LIDC"]
    rho_p8, p_p8 = _exact(x, pr); rho_p7, p_p7 = _exact(x[q], pr[q])
    rho_s8, p_s8 = _exact(x, su); rho_s7, p_s7 = _exact(x[q], su[q])
    assert abs(rho_p8 + 0.857) < 1e-3 and abs(p_p8 - 0.0107) < 1e-3
    VALUES["fig9_spearman"] = {"protocol_8": [round(rho_p8, 3), round(p_p8, 4)], "protocol_7": [round(rho_p7, 3), round(p_p7, 4)],
                               "supervision_8": [round(rho_s8, 3), round(p_s8, 4)], "supervision_7": [round(rho_s7, 3), round(p_s7, 4)]}

    fig = plt.figure(figsize=(W, 4.5))
    modm = {"CT": "s", "MRI": "o"}
    OFF = [{"brain-tumour t2": (8, 4), "LIDC-IDRI": (0, 8), "brain-growth": (0, 8), "prostate t2": (0, 8),
            "brain-tumour t3": (-14, 8), "brain-tumour t1": (0, 9), "kidney": (0, -9), "prostate t1": (-4, 9)},
           {"brain-tumour t2": (8, 5), "LIDC-IDRI": (0, 8), "brain-growth": (0, 8), "prostate t2": (6, 9),
            "brain-tumour t3": (-16, 8), "brain-tumour t1": (-26, -4), "kidney": (8, 8), "prostate t1": (-4, -9)}]
    for pi, (yv, ylab, rr, title) in enumerate([
            (pr, "Evaluation-target (protocol) effect", (rho_p8, p_p8, rho_p7, p_p7), "Protocol effect"),
            (su, "Supervision-target effect", (rho_s8, p_s8, rho_s7, p_s7), "Learning-design effect")]):
        ax = fig.add_axes([0.075 + pi * 0.50, 0.56, 0.40, 0.40])
        for i, (key, name, n, mod) in enumerate(TASKS):
            mk = "*" if key == "LIDC" else modm[mod]
            ax.scatter(x[i], yv[i], s=(14 + 5 * n) * (2.6 if key == "LIDC" else 1), marker=mk,
                       color=BLUE if key == "LIDC" else INK2, edgecolor="white", linewidth=0.5, zorder=4)
            ox, oy = OFF[pi][name]
            ax.annotate(name, xy=(x[i], yv[i]), xytext=(ox, oy), textcoords="offset points", ha="center", va="center",
                        fontsize=5.1, color=INK2)
        ax.text(0.98, 0.97, f"Spearman ρ, 8 tasks = {num(rr[0])} (exact p = {rr[1]:.3f})\n"
                f"7 QUBIQ tasks only = {num(rr[2])} (exact p = {rr[3]:.4f})" if rr[3] < 0.001 else
                f"Spearman ρ, 8 tasks = {num(rr[0])} (exact p = {rr[1]:.3f})\n7 QUBIQ tasks only = {num(rr[2])} (exact p = {rr[3]:.3f})",
                transform=ax.transAxes, ha="right", va="top", fontsize=5.2, color=INK2)
        ax.set_xlim(0.1, 1.02)
        ax.set_ylim(-1.5, 36 if pi == 0 else 21)
        ax.set_xlabel("Rater agreement index |V ≥ N| / |V ≥ 1|")
        ax.set_ylabel(f"{ylab}\n(Dice points, calibrated)")
        grid(ax, "both")
        tag(ax, "ab"[pi], x=-0.16, y=1.02)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker="s", ls="none", color=INK2, ms=4, label="QUBIQ, CT"),
                        Line2D([], [], marker="o", ls="none", color=INK2, ms=4, label="QUBIQ, MRI"),
                        Line2D([], [], marker="*", ls="none", color=BLUE, ms=7, label="LIDC-IDRI (this study)"),
                        Line2D([], [], marker="o", ls="none", color=INK3, ms=2.5, label="marker size grows with rater count (3–7)"),
                        Line2D([], [], color=BLUE, lw=1.2, label="c: union-trained (A)"),
                        Line2D([], [], color=ORANGE, lw=1.2, label="majority-trained (B)"),
                        Line2D([], [], color=VIOLET, lw=1.2, label="majority + soft (C)")],
               loc="upper center", bbox_to_anchor=(0.5, 0.478), ncol=7, fontsize=5.1, columnspacing=0.9, handletextpad=0.3)
    # family-level sensitivity text
    fr = fam["family_aggregated_5pt"]
    VALUES["fig9_family"] = fr

    # ---------------- c: Dice vs consensus level, eight tasks
    order = sorted(range(len(TASKS)), key=lambda i: x[i])
    gs = fig.add_gridspec(1, 8, left=0.065, right=0.995, top=0.355, bottom=0.08, wspace=0.42)
    armc = {"A": BLUE, "B": ORANGE, "C": VIOLET}
    for j, i in enumerate(order):
        key, name, n, mod = TASKS[i]
        ax = fig.add_subplot(gs[0, j])
        if key == "LIDC":
            cv = {a: [100 * T[t][str(k)]["dice_opt"] for k in (1, 2, 3, 4)] for a, t in (("A", "tgtA"), ("B", "tgtB"), ("C", "tgtC"))}
        else:
            cv = curves[key]["opt_thr"]
        ks = np.arange(1, n + 1) / n
        for a in "ABC":
            ax.plot(ks, cv[a], color=armc[a], lw=1.0, marker="o", ms=2.0, mec="white", mew=0.3, zorder=3)
        lo = min(min(v) for v in cv.values()); hi = max(max(v) for v in cv.values())
        pad = max(1.0, 0.12 * (hi - lo))
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(0.05, 1.05); ax.set_xticks([1 / n, 1], ["1/N", "N/N"], fontsize=5.0)
        ax.set_title(f"{name}\nindex {x[i]:.2f}", fontsize=5.4, pad=2)
        ax.tick_params(axis="y", labelsize=5.0)
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
        grid(ax)
        if j == 0:
            ax.set_ylabel("Dice (points)")
            tag(ax, "c", x=-0.75, y=1.15)
        VALUES.setdefault("fig9c", {})[name] = cv
    fig.text(0.53, 0.012, "Evaluation reference V ≥ k, as fraction of raters k/N (tasks ordered by agreement index)",
             ha="center", fontsize=6.0)
    fig.legend(handles=[Line2D([], [], color=armc[a], lw=1.2, label=l) for a, l in
                        (("A", "union-trained"), ("B", "majority-trained"), ("C", "majority + soft"))],
               loc="upper center", bbox_to_anchor=(0.5, 0.445), ncol=3, fontsize=5.6) if False else None
    VALUES["fig9_points"] = {TASKS[i][1]: {"agreement": round(x[i], 4), "protocol": round(pr[i], 2), "supervision": round(su[i], 2)}
                             for i in range(len(TASKS))}
    save(fig, "Fig9")




if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or [f"fig{i}" for i in range(1, 10)]
    for w in which:
        globals()[w]()

    def _conv(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    vfile = os.path.join(OUT, "figure_values_v23.json")
    old = json.load(open(vfile, encoding="utf-8")) if (os.path.exists(vfile) and len(which) < 9) else {}
    old.update(VALUES)
    with open(vfile, "w", encoding="utf-8") as f:
        json.dump(old, f, ensure_ascii=False, indent=1, default=_conv)
    print("values ->", vfile)
