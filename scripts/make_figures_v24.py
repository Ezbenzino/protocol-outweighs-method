# -*- coding: utf-8 -*-
"""make_figures_v24.py — the nine figures of manuscript v24, redrawn from the same released data as v23.

What changed relative to make_figures_v23.py (data handling is unchanged; every plotted number is the
same quantity computed by the same code path, and figure_values_v24.json is checked against v23):
  * one colour vocabulary for the whole paper, introduced in Fig. 1:
      - nested references G1..G4: ordinal magma ramp (light = union, dark = unanimous);
      - supervision arms A..F: fixed categorical colours (A blue, B vermillion, C purple, D green, E pink, F grey);
      - architectures: neutral greys (they are the null comparison);
      - data-protocol arms derived from B: B keeps its vermillion; + background / + translation / + both are
        light grey / dark grey / black with triangle / square / diamond markers;
      - effect categories (Figs 7-8): evaluation protocol black, data/training protocol mid grey,
        learning design light grey, noise floor shaded band.
    v23 reused the arm colours for effect categories and for the data-protocol arms.
  * no grid lines, no labels drawn across data, no mathtext sub/superscripts (Unicode instead), every glyph
    >= 5 pt at the final embedded width of 165 mm (figures are exported at their final size, no tight bbox).
  * per-panel fixes: sorted gallery (Fig. 2a), transposed threshold map with horizontal values (Fig. 3c),
    full data range in Fig. 4b (v23 clipped 17 cases), direct-label/legend placement outside data
    (Figs 2, 4, 5, 6, 8, 9), merged arm visually marked as the confound-free estimate (Figs 6, 7).
  * render-time panel-alignment gate, PDF glyph-size audit and PDF collision audit (nature-figure QA).

Inputs: outputs/figures/v23/source_data/* and the analysis JSON files listed in make_figures_v23.py.
Outputs: outputs/figures/v24/Fig1..Fig9 (.pdf .svg .png 600 dpi .tiff 600 dpi), figure_values_v24.json,
         qa/FigN.alignment.json.
Usage (repository root):  python scripts/make_figures_v24.py            # or: ... fig3 fig5
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
from matplotlib import patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle
from matplotlib.transforms import ScaledTranslation, blended_transform_factory
from scipy import stats
from scipy.stats import rankdata, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("FIG_ROOT") or os.path.dirname(HERE)
OUT = os.environ.get("FIG_OUT") or os.path.join(ROOT, "outputs", "figures", "v24")
SD = os.environ.get("FIG_SD") or os.path.join(ROOT, "outputs", "figures", "v23", "source_data")
QA = os.path.join(OUT, "qa")
os.makedirs(QA, exist_ok=True)
sys.path.insert(0, os.environ.get("NATURE_FIGURE_SCRIPTS", HERE))
try:      # nature-figure render-time alignment gate (blocking when available)
    from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402
except ImportError:      # the released script stays runnable without the QA module
    require_matplotlib_panel_alignment = None
    print("warning: audit_panel_alignment.py not found; panel-alignment gate skipped")

W = 6.5                                  # final embedded width in the Word manuscript (16.51 cm)
MINUS = "−"

# ------------------------------------------------------------------ colour vocabulary (see module docstring)
INK, INK2, INK3, LIGHT, BAND = "#1a1a1a", "#4d4d4d", "#8c8c8c", "#d4d4d4", "#ececec"
ARM_C = {"tgtA": "#0072B2", "tgtB": "#D55E00", "tgtC": "#6A3D9A", "tgtD": "#009E73", "tgtE": "#CC79A7",
         "tgtF": "#8c8c8c",
         "archR18": "#7a7a7a", "archCNX": "#7a7a7a", "archPVT": "#7a7a7a", "archPU1e3": "#7a7a7a",
         "tgtBshift": "#595959", "bg_maj": "#a6a6a6", "tgtBshift_bg": "#000000"}
ARM_M = {"tgtB": "o", "bg_maj": "^", "tgtBshift": "s", "tgtBshift_bg": "D"}
ARM_L = {"tgtA": "A union", "tgtB": "B majority", "tgtC": "C majority + soft", "tgtD": "D soft only",
         "tgtE": "E SVLS", "tgtF": "F SoftSeg", "archR18": "ResNet-18", "archCNX": "ConvNeXt-T",
         "archPVT": "PVT v2-b1", "archPU1e3": "Plain U-Net", "tgtBshift": "B + translation",
         "bg_maj": "B + background", "tgtBshift_bg": "B + both"}
CAT_C = {"protocol": "#1a1a1a", "data": "#7a7a7a", "method": "#c8c8c8", "noise": "#ececec"}
_mag = plt.get_cmap("magma")
REF_C = [mpl.colors.to_hex(_mag(x)) for x in (0.86, 0.66, 0.45, 0.24)]
VOTE_CMAP = mpl.colors.ListedColormap(["#000000"] + [mpl.colors.to_hex(_mag(x)) for x in (0.24, 0.45, 0.66, 0.86)])
DICE_CMAP = mpl.colors.LinearSegmentedColormap.from_list("dice", ["#f7f7f7", "#bdbdbd", "#636363", "#1a1a1a"])
THR_CMAP = mpl.colors.LinearSegmentedColormap.from_list("thr", ["#fff7e6", "#f2cf8a", "#c98a1b", "#7a4a00"])
SUB = str.maketrans("0123456789k", "₀₁₂₃₄₅₆₇₈₉ₖ")


def G(k):
    return "G" + str(k).translate(SUB)


mpl.rcParams.update({
    "font.family": ["Arial", "Liberation Sans", "DejaVu Sans"], "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 6.5, "axes.labelsize": 6.5, "axes.titlesize": 6.5,
    "xtick.labelsize": 6.0, "ytick.labelsize": 6.0, "legend.fontsize": 6.0,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.45, "ytick.minor.width": 0.45,
    "xtick.major.size": 2.4, "ytick.major.size": 2.4, "xtick.minor.size": 1.4, "ytick.minor.size": 1.4,
    "xtick.major.pad": 1.8, "ytick.major.pad": 1.8, "axes.labelpad": 2.0, "axes.titlepad": 3.0,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "axes.labelcolor": INK, "text.color": INK, "axes.grid": False,
    "legend.frameon": False, "legend.handlelength": 1.4, "legend.borderaxespad": 0.2, "legend.labelspacing": 0.3,
    "figure.dpi": 120, "savefig.dpi": 600, "savefig.bbox": None,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "axes.unicode_minus": True,
})

VALUES = {}


def J(*p):
    with open(os.path.join(ROOT, *p), encoding="utf-8") as f:
        return json.load(f)


def num(x, nd=2):
    return f"{x:.{nd}f}".replace("-", MINUS)


def sgn(x, nd=2):
    return f"{x:+.{nd}f}".replace("-", MINUS)


def sci(p):
    m, e = f"{p:.1e}".split("e")
    sup = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    return f"{m} × 10{str(int(e)).translate(sup)}"


def halo(lw=1.6, c="white"):
    return [pe.Stroke(linewidth=lw, foreground=c), pe.Normal()]


def boot_mean_ci(x, B=5000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)])
    return np.percentile(m, [2.5, 97.5])


def plabel(ax, s, dx=-24, dy=5):
    """Bold lowercase panel label at a fixed physical offset from the axes' top-left corner."""
    tr = ax.transAxes + ScaledTranslation(dx / 72, dy / 72, ax.figure.dpi_scale_trans)
    ax.text(0, 1, s, transform=tr, fontsize=8, fontweight="bold", va="bottom", ha="left", color=INK)


def flabel(fig, s, x, y):
    fig.text(x, y, s, fontsize=8, fontweight="bold", va="top", ha="left", color=INK)


def save(fig, name, row_groups=None, column_groups=None, exclude=(), exemptions=()):
    """Alignment gate (blocks on FAIL/WARN), then export at final size."""
    kw = {}
    if row_groups is not None:
        kw["row_groups"] = row_groups
    if column_groups is not None:
        kw["column_groups"] = column_groups
    rep = {"verdict": "SKIPPED"} if require_matplotlib_panel_alignment is None else require_matplotlib_panel_alignment(
        fig, json_out=os.path.join(QA, f"{name}.alignment.json"), overlay_svg=os.path.join(QA, f"{name}.alignment.svg"),
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True, exclude_axes=exclude, exemptions=list(exemptions), **kw)
    VALUES.setdefault("_qa_alignment", {})[name] = rep.get("verdict") or rep.get("status")
    fig.savefig(os.path.join(OUT, f"{name}.pdf"))
    fig.savefig(os.path.join(OUT, f"{name}.svg"))
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=600)
    fig.savefig(os.path.join(OUT, f"{name}.tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(f"  -> {name}  alignment: {VALUES['_qa_alignment'][name]}")


# ------------------------------------------------------------------ loaders (identical to v23)
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


def thr_test():
    return J("outputs", "analysis_test", "symmetric_analysis_applied.json")["table1"]


def pc_cv(arm, k, thr):
    d = case_cv()
    s = d[(d.arm == arm) & (d.k == k) & np.isclose(d.threshold, thr)]
    return s.set_index(["fold", "case_id"]).dice


def pc_test(arm, k, thr, stratum="all"):
    d = case_test()
    s = d[(d.arm == arm) & (d.k == k) & np.isclose(d.threshold, thr) & (d.stratum == stratum)]
    return s.set_index("case_id").dice


def logit_axis(ax, lim=(0.0008, 0.9992), majors=(0.01, 0.5, 0.99)):
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    ax.set_xscale("logit")
    ax.xaxis.set_major_locator(FixedLocator(list(majors)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(FixedLocator([0.002, 0.005, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7,
                                             0.8, 0.9, 0.95, 0.98, 0.995, 0.998]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(*lim)


def _paired(a, b, k, test):
    if test:
        Tt = thr_test()
        x, y = pc_test(a, k, Tt[a][str(k)]["thr"]), pc_test(b, k, Tt[b][str(k)]["thr"])
    else:
        T = thr_cv()
        x, y = pc_cv(a, k, T[a][str(k)]["thr"]), pc_cv(b, k, T[b][str(k)]["thr"])
    x, y = x.align(y, join="inner")
    return 100 * (x - y).values


def ref_key(ax, x0, y0, labels=("V ≥ 1", "V ≥ 2", "V ≥ 3", "V ≥ 4"), ks=(1, 2, 3, 4), dx=0.085, fig=None, lw=1.6):
    """Figure-level colour key for the nested references (figure coordinates)."""
    for i, k in enumerate(ks):
        x = x0 + i * dx
        fig.add_artist(Line2D([x, x + 0.022], [y0, y0], color=REF_C[k - 1], lw=lw, transform=fig.transFigure))
        fig.text(x + 0.027, y0, labels[i], va="center", ha="left", fontsize=6.0, color=INK)


# ====================================================================== Fig. 1
def _box(fig, x, y, w, h, fc="#f7f7f7", ec=INK3, lw=0.5, r=0.006):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", transform=fig.transFigure,
                       fc=fc, ec=ec, lw=lw, zorder=-1)
    fig.add_artist(p)
    return p


def _arrow(fig, x0, y0, x1, y1, c=INK2, lw=0.7, ms=6):
    fig.add_artist(FancyArrowPatch((x0, y0), (x1, y1), transform=fig.transFigure, arrowstyle="-|>",
                                   mutation_scale=ms, color=c, lw=lw, shrinkA=0, shrinkB=0))


def fig1():
    z = np.load(os.path.join(SD, "gallery_candidates.npz"))
    gi = 30                                   # 8.1 mm spiculated nodule, LIDC-IDRI-0608 (cross-validation pool)
    img, vote = z["imgs"][gi].astype(float), z["votes"][gi]
    c = 32; h = 22
    img, vote = img[c - h:c + h, c - h:c + h], vote[c - h:c + h, c - h:c + h]
    H = 4.85
    fig = plt.figure(figsize=(W, H))

    # ---------------- a: one vote map, four nested references
    flabel(fig, "a", 0.0, 0.995)
    fig.text(0.035, 0.99, "One vote map, four nested references", fontsize=6.8, va="top")
    y0, hh = 0.77, 0.155
    ww = hh * H / W
    gap = 0.03
    xs = [0.035, 0.035 + ww + gap]
    ax = fig.add_axes([xs[0], y0, ww, hh]); ax.set_gid("a-ct")
    ax.imshow(img, cmap="gray", vmin=0, vmax=255); ax.axis("off")
    ax.set_title("CT, lung window", fontsize=6.0, pad=2)
    ax = fig.add_axes([xs[1], y0, ww, hh]); ax.set_gid("a-vote")
    ax.imshow(vote, cmap=VOTE_CMAP, vmin=-0.5, vmax=4.5, interpolation="nearest"); ax.axis("off")
    ax.set_title("Vote map V(x), 4 readers", fontsize=6.0, pad=2)
    _arrow(fig, xs[0] + ww + 0.005, y0 + hh / 2, xs[1] - 0.005, y0 + hh / 2)
    # discrete key for V (and p = V/4)
    kx, ky, kw, kh = xs[1], y0 - 0.038, ww / 5, 0.016
    for v in range(5):
        fig.add_artist(Rectangle((kx + v * kw, ky), kw, kh, transform=fig.transFigure,
                                 fc=VOTE_CMAP(v), ec="white", lw=0.4))
        fig.text(kx + (v + 0.5) * kw, ky - 0.006, f"{v}", ha="center", va="top", fontsize=5.4, color=INK)
    fig.text(kx - 0.006, ky + kh / 2, "V", ha="right", va="center", fontsize=5.6, color=INK)
    x = xs[1] + ww + gap
    _arrow(fig, xs[1] + ww + 0.005, y0 + hh / 2, x - 0.005, y0 + hh / 2)
    gx = []
    for k in (1, 2, 3, 4):
        ax = fig.add_axes([x, y0, ww, hh]); ax.set_gid(f"a-G{k}")
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        m = (vote >= k).astype(float)
        ax.imshow(np.ma.masked_where(m == 0, m), cmap=mpl.colors.ListedColormap([REF_C[k - 1]]), alpha=0.55, vmin=0, vmax=1)
        ax.contour(m, [0.5], colors=[REF_C[k - 1]], linewidths=0.8)
        ax.axis("off")
        ax.set_title(f"{G(k)}: V ≥ {k}  ({int(m.sum())} px)", fontsize=6.0, pad=2)
        gx.append(x)
        x += ww + 0.012
    ax = fig.add_axes([x + 0.014, y0, ww, hh]); ax.set_gid("a-p")
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    p = np.ma.masked_where(vote == 0, vote / 4.0)
    ax.imshow(p, cmap=mpl.colors.ListedColormap([VOTE_CMAP(v) for v in range(1, 5)]), vmin=0.125, vmax=1.125, alpha=0.85,
              interpolation="nearest")
    ax.axis("off")
    ax.set_title("p = V/4 (soft target)", fontsize=6.0, pad=2)
    fig.text((gx[0] + gx[-1] + ww) / 2, y0 - 0.012,
             "G₁ ⊇ G₂ ⊇ G₃ ⊇ G₄  by construction:  union · majority · high consensus · unanimous",
             fontsize=5.6, color=INK2, va="top", ha="center")

    # ---------------- b: design matrix
    flabel(fig, "b", 0.0, 0.672)
    fig.text(0.035, 0.667, "Thirteen trained configurations; each differs from base arm B only in the shaded cells",
             fontsize=6.8, va="top")
    cols = ["Dice term", "Pixel term", "Encoder", "Transl.", "Backgr.", "Runs", "CV", "Test", "Offset", "FOV"]
    cx = [0.20, 0.262, 0.330, 0.387, 0.433, 0.475, 0.512, 0.546, 0.585, 0.622]
    rows = [
        ("Supervision target", None, None),
        ("A  union", "tgtA", ["G₁", "G₁", "R34", "", "", "5", 1, 1, 1, 0]),
        ("B  majority (base)", "tgtB", ["G₂", "G₂", "R34", "", "", "5", 1, 1, 1, 1]),
        ("C  majority + soft", "tgtC", ["G₂", "p", "R34", "", "", "5", 1, 1, 1, 0]),
        ("D  soft only", "tgtD", ["p", "p", "R34", "", "", "5", 1, 1, 1, 0]),
        ("E  SVLS", "tgtE", ["G₂", "SVLS(G₂)", "R34", "", "", "5", 1, 1, 1, 0]),
        ("F  SoftSeg*", "tgtF", ["—", "p (AW)", "R34", "", "", "5", 1, 1, 1, 0]),
        ("Architecture (target B)", None, None),
        ("ResNet-18", "archR18", ["G₂", "G₂", "R18", "", "", "5", 1, 1, 1, 0]),
        ("ConvNeXt-Tiny", "archCNX", ["G₂", "G₂", "CNX-T", "", "", "5", 1, 1, 1, 0]),
        ("PVT v2-b1", "archPVT", ["G₂", "G₂", "PVT", "", "", "5", 1, 1, 1, 0]),
        ("Plain U-Net", "archPU1e3", ["G₂", "G₂", "none", "", "", "5", 1, 1, 1, 0]),
        ("Data protocol (target B)", None, None),
        ("+ translation ±32 px", "tgtBshift", ["G₂", "G₂", "R34", "●", "", "5", 1, 1, 1, 1]),
        ("+ background 50%", "bg_maj", ["G₂", "G₂", "R34", "", "●", "2", 1, 1, 0, 1]),
        ("+ both", "tgtBshift_bg", ["G₂", "G₂", "R34", "●", "●", "2", 1, 1, 0, 1]),
        ("Noise floor", None, None),
        ("A, B × 3 seeds", None, ["", "", "R34", "", "", "12", 1, 0, 0, 0]),
    ]
    ytop, dy = 0.625, 0.0285
    for xi, cname in zip(cx, cols):
        fig.text(xi, ytop + 0.005, cname, ha="center", va="bottom", fontsize=5.6, color=INK2)
    fig.add_artist(Line2D([0.035, 0.645], [ytop + 0.002, ytop + 0.002], transform=fig.transFigure, color=INK3, lw=0.5))
    y = ytop - 0.016
    base_vals = ["G₂", "G₂", "R34", "", "", "5"]
    for name, arm, vals in rows:
        if vals is None:
            fig.text(0.035, y, name, fontsize=6.0, fontweight="bold", va="center", color=INK)
            y -= dy * 0.95
            continue
        is_base = arm == "tgtB"
        if is_base:
            fig.add_artist(Rectangle((0.035, y - dy / 2), 0.61, dy, transform=fig.transFigure, fc="#f2f2f2", ec="none", zorder=0))
        col = ARM_C.get(arm, INK) if arm and arm.startswith("tgt") and arm in ("tgtA", "tgtB", "tgtC", "tgtD", "tgtE", "tgtF") else INK
        if arm in ("tgtA", "tgtB", "tgtC", "tgtD", "tgtE", "tgtF"):
            fig.add_artist(Line2D([0.043], [y], transform=fig.transFigure, marker="s", ms=3.6, ls="none", mfc=col, mec=col))
        elif arm in ARM_M:
            fig.add_artist(Line2D([0.043], [y], transform=fig.transFigure, marker=ARM_M[arm], ms=3.4, ls="none",
                                  mfc=ARM_C[arm], mec=ARM_C[arm]))
        fig.text(0.052, y, name, fontsize=6.0, va="center", color=INK)
        for j, (xi, v) in enumerate(zip(cx, vals)):
            if j >= 6:
                fig.add_artist(Line2D([xi], [y], transform=fig.transFigure, marker="o", ls="none",
                                      ms=3.4 if v else 2.8, mfc=INK2 if v else "white", mec=INK2 if v else INK3, mew=0.5))
                continue
            diff = (not is_base) and arm is not None and j < 5 and v != base_vals[j]
            if diff:
                cw = 0.066 if j == 1 else 0.054
                fig.add_artist(Rectangle((xi - cw / 2, y - dy / 2 + 0.0025), cw, dy - 0.005, transform=fig.transFigure,
                                         fc="#dcdcdc", ec="none", zorder=0))
            if v == "●":
                fig.add_artist(Line2D([xi], [y], transform=fig.transFigure, marker="o", ls="none", ms=3.4, mfc=INK, mec=INK))
            else:
                fig.text(xi, y, v, fontsize=6.0, ha="center", va="center", color=INK, fontweight="bold" if diff else "normal")
        y -= dy
    fig.text(0.035, y + 0.006,
             "Shaded: where a configuration differs from base arm B. R34, ImageNet ResNet-34 encoder; p = V/4;\n"
             "AW, Adaptive Wing loss. *F changes target, activation and loss together and is reported outside\n"
             "the factorial design. Dots at right: where each configuration was evaluated. 71 LIDC training runs;\n"
             "all share optimiser, schedule, splits and the Dice + BCE objective.",
             fontsize=5.4, color=INK2, va="top", linespacing=1.3)

    # ---------------- c: evaluation expansion
    flabel(fig, "c", 0.668, 0.672)
    fig.text(0.70, 0.667, "One forward pass, every protocol", fontsize=6.8, va="top")
    _box(fig, 0.675, 0.405, 0.32, 0.235, fc="#fafafa")
    fig.text(0.688, 0.61, "References", fontsize=6.0, color=INK2, va="center")
    for k in (1, 2, 3, 4):
        fig.add_artist(mpl.patches.Circle((0.783 + (k - 1) * 0.0045, 0.61), 0.016 - (k - 1) * 0.0033,
                                          transform=fig.transFigure, fc=REF_C[k - 1], ec="white", lw=0.4))
    fig.text(0.815, 0.61, "× 4  (V ≥ 1 … V ≥ 4)", fontsize=6.0, va="center")
    fig.text(0.688, 0.563, "Thresholds", fontsize=6.0, color=INK2, va="center")
    tax = fig.add_axes([0.77, 0.554, 0.105, 0.018]); tax.set_gid("c-thr")
    tax.set_xscale("logit")
    tt = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75,
          0.8, 0.85, 0.9, 0.925, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998, 0.999]
    tax.vlines(tt, 0, 1, color=INK2, lw=0.35); tax.set_xlim(0.0008, 0.9992); tax.axis("off")
    fig.text(0.882, 0.563, "× 33, 0.001–0.999", fontsize=6.0, va="center")
    fig.text(0.688, 0.515, "Operating\npoints", fontsize=6.0, color=INK2, va="center", linespacing=1.1)
    fig.text(0.77, 0.515, "fixed 0.5 · calibrated on validation\n· area-matched to G₂", fontsize=6.0, va="center", linespacing=1.2)
    fig.text(0.688, 0.464, "Extent", fontsize=6.0, color=INK2, va="center")
    for i, s in enumerate((0.010, 0.015, 0.020, 0.030, 0.040)):
        g_ = ["#1a1a1a", "#4d4d4d", "#7a7a7a", "#a6a6a6", "#c8c8c8"][i]
        fig.add_artist(Rectangle((0.79 - s / 2, 0.464 - s * W / H / 2), s, s * W / H, transform=fig.transFigure,
                                 fill=False, ec=g_, lw=0.6))
    fig.text(0.825, 0.464, "patch · windows 128–512\n· slice · volume  (× 8)", fontsize=6.0, va="center", linespacing=1.2)
    fig.text(0.688, 0.42, "Offset", fontsize=6.0, color=INK2, va="center")
    for dd, a_ in ((0, 1.0), (0.007, 0.65), (0.014, 0.35)):
        fig.add_artist(Rectangle((0.772 + dd, 0.409), 0.02, 0.02 * W / H, transform=fig.transFigure, fill=False,
                                 ec=INK, lw=0.6, alpha=a_))
    fig.text(0.825, 0.42, "d = 0–56 px × 6 directions", fontsize=6.0, va="center")

    # ---------------- d: data, splits and replication
    flabel(fig, "d", 0.668, 0.385)
    fig.text(0.70, 0.38, "Data, splits and replication", fontsize=6.8, va="top")
    _box(fig, 0.675, 0.288, 0.32, 0.068, fc="#f7f7f7", ec=INK3)
    fig.text(0.685, 0.322, "LIDC-IDRI: 1018 series parsed, 1010 enter the study;\ncase-level split; lesion = V ≥ 2 component ≥ 3 mm",
             fontsize=5.8, va="center", linespacing=1.25)
    _box(fig, 0.675, 0.152, 0.153, 0.108, fc="white", ec=INK3)
    fig.text(0.7515, 0.206, "Cross-validation\n808 cases, 5 folds\n634 contributing\n8471 instances", ha="center", va="center",
             fontsize=5.8, linespacing=1.25)
    _box(fig, 0.842, 0.152, 0.153, 0.108, fc="white", ec=INK, lw=0.9)
    fig.text(0.9185, 0.206, "Held-out test\n202 cases\n158 contributing\n2062 instances",
             ha="center", va="center", fontsize=5.8, linespacing=1.25)
    _arrow(fig, 0.7515, 0.288, 0.7515, 0.262); _arrow(fig, 0.9185, 0.288, 0.9185, 0.262)
    fig.text(0.835, 0.143, "thresholds selected on validation folds, imported unchanged", ha="center", va="top",
             fontsize=5.6, color=INK2)
    _box(fig, 0.675, 0.022, 0.32, 0.092, fc="#f7f7f7", ec=INK3)
    fig.text(0.685, 0.068, "External: QUBIQ 2021, 7 tasks from 4 collections\n(brain growth, brain tumour × 3, kidney, prostate × 2;\n"
             "3–7 raters; CT and MRI) × 3 targets × 5 folds = 105 runs", fontsize=5.8, va="center", linespacing=1.25)
    strip = ["a-ct", "a-vote", "a-G1", "a-G2", "a-G3", "a-G4", "a-p"]
    save(fig, "Fig1", row_groups=[{"id": "a-strip", "panels": strip}], column_groups=[],
         exclude=[a for a in fig.axes if a.get_gid() == "c-thr"],
         exemptions=[{"panels": strip, "checks": ["horizontal-gutter"],
                      "reason": "wider gaps carry the derivation arrows (CT -> vote map -> references) and separate p = V/4"}])


# ====================================================================== Fig. 2
GALLERY = [4, 18, 44, 29, 30, 57, 50, 67]     # v23 selection, now ordered by diameter within each stratum


def fig2():
    z = np.load(os.path.join(SD, "gallery_candidates.npz"))
    meta = json.load(open(os.path.join(SD, "gallery_candidates.json")))
    ic = pd.read_csv(os.path.join(SD, "instances_cv.csv")); ic["set"] = "Cross-validation"
    it = pd.read_csv(os.path.join(SD, "instances_test.csv")); it["set"] = "Held-out test"
    inst = pd.concat([ic, it], ignore_index=True)
    for k in (2, 3, 4):
        inst[f"r{k}"] = inst[f"gt_area_v{k}"] / inst.gt_area_v1
    inst["stratum"] = np.where(inst.diameter_mm < 5, "3–5", np.where(inst.diameter_mm < 10, "5–10", "10–30"))

    H = 3.95
    fig = plt.figure(figsize=(W, H))
    # ---------------- a: gallery
    gs_top = fig.add_gridspec(1, 8, left=0.01, right=0.99, top=0.865, bottom=0.665, wspace=0.06)
    groups = {"3–5 mm": (0, 1), "5–10 mm": (2, 4), "10–30 mm": (5, 7)}
    axes = []
    for j, gi in enumerate(GALLERY):
        ax = fig.add_subplot(gs_top[0, j]); ax.set_gid(f"a{j + 1}"); axes.append(ax)
        im, v = z["imgs"][gi].astype(float), z["votes"][gi]
        h = 22 if j < 2 else (26 if j < 5 else 32)
        c = im.shape[0] // 2
        im, v = im[c - h:c + h, c - h:c + h], v[c - h:c + h, c - h:c + h]
        ax.imshow(im, cmap="gray", vmin=0, vmax=255, interpolation="bilinear")
        for k in (1, 2, 3, 4):
            m = (v >= k).astype(float)
            if m.sum():
                cs = ax.contour(m, [0.5], colors=[REF_C[k - 1]], linewidths=0.85)
                cs.set_path_effects(halo(1.7, "#000000"))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color("#bdbdbd"); sp.set_linewidth(0.5)
        a = meta[gi]["a"]
        ax.text(0.05, 0.95, f"{meta[gi]['diam']:.1f} mm", transform=ax.transAxes, color="white", fontsize=6.0,
                va="top", ha="left", path_effects=halo(1.4, "#000000"))
        lab = "no unanimous pixel" if a[3] == 0 else f"|G₄|/|G₁| = {a[3] / a[0]:.2f}"
        ax.text(0.5, -0.05, f"{a[0]}/{a[1]}/{a[2]}/{a[3]} px\n{lab}", transform=ax.transAxes, ha="center",
                va="top", fontsize=5.6, color=INK2, linespacing=1.15)
    for name, (j0, j1) in groups.items():
        x0 = axes[j0].get_position().x0; x1 = axes[j1].get_position().x1
        fig.add_artist(Line2D([x0 + 0.004, x1 - 0.004], [0.885, 0.885], color=INK3, lw=0.6))
        fig.text((x0 + x1) / 2, 0.893, name, ha="center", va="bottom", fontsize=6.3, color=INK)
    flabel(fig, "a", 0.0, 0.995)
    fig.text(0.035, 0.99, "Reference contours:", fontsize=6.0, va="top", color=INK2)
    ref_key(None, 0.165, 0.977, fig=fig)
    VALUES["fig2_gallery"] = [dict(meta[g], shown=True) for g in GALLERY]

    # ---------------- b-e: cohort statistics (one row, shared vertical extent)
    gs = fig.add_gridspec(1, 4, left=0.075, right=0.99, top=0.47, bottom=0.115, wspace=0.52,
                          width_ratios=[1.05, 1.0, 1.15, 0.85])
    ax = fig.add_subplot(gs[0, 0]); ax.set_gid("b")
    for k in (2, 3, 4):
        x = np.sort(inst[f"r{k}"].values)
        yv = np.arange(1, len(x) + 1) / len(x)
        ax.step(x, yv, where="post", color=REF_C[k - 1], lw=1.2)
    p4 = (inst.r4 == 0).mean(); p3 = (inst.r3 == 0).mean()
    ax.plot([0], [p4], "o", ms=3.0, color=REF_C[3], mec="white", mew=0.5, zorder=5, clip_on=False)
    ax.text(0.035, p4 - 0.045, f"{100 * p4:.1f}%", fontsize=5.6, color=INK2, va="center", ha="left")
    ax.set_xlim(-0.02, 1.0); ax.set_ylim(0, 1.0)
    ax.set_xlabel("|Gₖ|/|G₁|, area relative to union")
    ax.set_ylabel("Cumulative fraction of instances")
    plabel(ax, "b", dx=-27)

    bx = fig.add_subplot(gs[0, 1]); bx.set_gid("c")
    strata = ["3–5", "5–10", "10–30"]
    xs = np.arange(3); w = 0.19
    out_c = {}
    for i, (k, setname, hatch) in enumerate([(3, "Cross-validation", None), (3, "Held-out test", "//////"),
                                             (4, "Cross-validation", None), (4, "Held-out test", "//////")]):
        vals = []
        for st in strata:
            s = inst[(inst.stratum == st) & (inst.set == setname)]
            vals.append(100 * (s[f"gt_area_v{k}"] == 0).mean())
        off = (i - 1.5) * (w + 0.02)
        if hatch is None:
            bx.bar(xs + off, vals, w, color=REF_C[k - 1], lw=0, zorder=3)
        else:          # held-out test: open bars (no hatch patterns; they confuse PDF geometry audits)
            bx.bar(xs + off, vals, w * 0.86, facecolor="white", edgecolor=REF_C[k - 1], lw=0.9, zorder=3)
        out_c[f"G{k}_{setname}"] = [round(v, 1) for v in vals]
    n_by = [int((inst.stratum == st).sum()) for st in strata]
    bx.set_xticks(xs, [f"{s}\n({n})" for s, n in zip(strata, n_by)], fontsize=5.6)
    bx.set_xlabel("Diameter, mm (instances)")
    bx.set_ylabel("Empty reference (% of instances)")
    bx.set_ylim(0, 100)
    bx.legend(handles=[Patch(facecolor=REF_C[2], label="V ≥ 3 empty, CV"), Patch(facecolor=REF_C[3], label="V ≥ 4 empty, CV"),
                       Patch(facecolor="white", edgecolor=INK2, lw=0.9, label="Open: held-out test")],
              loc="upper right", bbox_to_anchor=(1.02, 1.03), fontsize=5.6, handlelength=1.0, handleheight=0.8)
    plabel(bx, "c", dx=-27)
    VALUES["fig2_empty_by_stratum_pct"] = out_c

    cx = fig.add_subplot(gs[0, 2]); cx.set_gid("d")
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
        cx.plot(mids, mu, color=REF_C[k - 1], lw=1.2, marker="o", ms=2.6, mec="white", mew=0.4, zorder=4)
        VALUES[f"fig2_mean_ratio_r{k}_by_diam_bin"] = {f"{lo}-{hi}": round(v, 3) for lo, hi, v in zip(bins[:-1], bins[1:], mu)}
        cx.text(mids[-1] * 1.06, mu[-1], f"V ≥ {k}", va="center", ha="left", fontsize=5.6, color=REF_C[k - 1])
    cx.set_xscale("log")
    cx.set_xticks([3, 5, 10, 20, 30], ["3", "5", "10", "20", "30"])
    cx.minorticks_off()
    cx.set_xlim(2.9, 42)
    cx.set_ylim(0, 1.0)
    cx.set_xlabel("Equivalent diameter (mm)")
    cx.set_ylabel("Mean |Gₖ|/|G₁| (95% CI)")
    plabel(cx, "d", dx=-27)
    rho = pd.Series(d).corr(inst.r4, method="spearman")
    VALUES["fig2_spearman_diam_r4"] = round(float(rho), 3)

    ex = fig.add_subplot(gs[0, 3]); ex.set_gid("e")
    cnt = inst.groupby("case_id").size().values
    ex.hist(cnt, bins=np.arange(0, 82, 4), color=INK3, edgecolor="white", lw=0.4, zorder=3)
    ex.axvline(np.median(cnt), color=INK, lw=0.8, ls=(0, (3, 2)), zorder=4)
    ex.set_ylim(0, 245)
    ex.text(0.97, 0.97, f"median {np.median(cnt):.0f} (dashed)\nmaximum {cnt.max()}", transform=ex.transAxes,
            fontsize=5.6, va="top", ha="right", color=INK2)
    ex.set_xlabel("Instances per case")
    ex.set_ylabel(f"Cases (n = {len(cnt)})")
    plabel(ex, "e", dx=-27)
    VALUES["fig2_instances_per_case"] = {"median": float(np.median(cnt)), "max": int(cnt.max()), "n_cases": int(len(cnt))}
    VALUES["fig2_pct_no_unanimous"] = {"instances": round(100 * p4, 1), "G3_empty_instances": round(100 * p3, 1)}
    save(fig, "Fig2", row_groups=[{"id": "gallery", "panels": [f"a{j}" for j in range(1, 9)]},
                                  {"id": "stats", "panels": ["b", "c", "d", "e"]}],
         column_groups=[],
         exemptions=[{"panels": ["b", "c", "d", "e"], "checks": ["panel-width"],
                      "reason": "unequal width ratios chosen for content (cumulative curve, bars, trend, histogram)"}])


# ====================================================================== Fig. 3
def fig3():
    T = thr_cv(); Tt = thr_test()
    cv = case_cv()
    curves_src = pd.read_csv(os.path.join(ROOT, "outputs", "figures", "v22", "source_data", "cv_threshold_curves.csv"))
    R = J("outputs", "figures", "v22", "source_data", "cv_reference_areas.json")
    T5 = J("outputs", "analysis_full", "symmetric_analysis.json")["table5"]

    H = 5.7
    fig = plt.figure(figsize=(W, H))
    # ---------------- a: threshold curves (CV)
    gsa = fig.add_gridspec(1, 4, left=0.085, right=0.99, top=0.955, bottom=0.755, wspace=0.12)
    arms = ["tgtA", "tgtB", "tgtC", "tgtD"]
    g = cv[cv.arm.isin(arms)].groupby(["arm", "k", "threshold", "fold"]).dice.mean().groupby(["arm", "k", "threshold"]).mean()
    va = {}
    for j, k in enumerate((1, 2, 3, 4)):
        ax = fig.add_subplot(gsa[0, j]); ax.set_gid(f"a{k}")
        ax.axvline(0.5, color=INK3, lw=0.6, ls=(0, (3, 2)), zorder=1)
        for a in arms:
            s = g.loc[(a, k)]
            ax.plot(s.index, s.values, color=ARM_C[a], lw=1.1, zorder=3)
            t = T[a][str(k)]["thr"]
            v = float(s.loc[np.isclose(s.index, t)].iloc[0])
            assert abs(v - T[a][str(k)]["dice_opt"]) < 1e-6
            ax.plot([t], [v], "o", ms=3.4, color=ARM_C[a], mec="white", mew=0.6, zorder=5, clip_on=False)
            va[f"{a}_G{k}"] = {"thr": t, "dice": round(v, 4)}
        logit_axis(ax)
        ax.set_ylim(0, 0.92)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
        ax.set_title(f"Scored against {G(k)} (V ≥ {k})", fontsize=6.5)
        if j:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("Dice (cross-validation)")
            plabel(ax, "a", dx=-30)
    fig.text(0.5375, 0.715, "Binarisation threshold (logit scale)", ha="center", va="top", fontsize=6.5)
    h = [Line2D([], [], color=ARM_C[a], lw=1.4, label=ARM_L[a]) for a in arms] + \
        [Line2D([], [], ls="none", marker="o", ms=3.4, color=INK3, mec="white", label="Calibrated optimum"),
         Line2D([], [], color=INK3, lw=0.6, ls=(0, (3, 2)), label="Fixed 0.5")]
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5375, 0.695), ncol=6, fontsize=6.0,
               handlelength=1.3, columnspacing=1.1)
    VALUES["fig3a"] = va

    # ---------------- b-e: asymmetric grid; c spans both rows on the right
    gs = fig.add_gridspec(2, 3, left=0.105, right=0.865, top=0.60, bottom=0.075, wspace=0.34, hspace=0.62,
                          width_ratios=[1, 1, 0.60], height_ratios=[1, 1.15])
    rows = ["tgtA", "tgtB", "tgtC", "tgtD", "tgtE"]
    short = {"tgtA": "A union", "tgtB": "B majority", "tgtC": "C maj. + soft", "tgtD": "D soft only", "tgtE": "E SVLS"}
    for j, (key, title) in enumerate([("dice_05", "Fixed threshold 0.5"), ("dice_opt", "Calibrated threshold")]):
        ax = fig.add_subplot(gs[0, j]); ax.set_gid(f"b{j + 1}")
        M = np.array([[100 * T[a][str(k)][key] for k in (1, 2, 3, 4)] for a in rows])
        ax.imshow(M, cmap=DICE_CMAP, vmin=55, vmax=92, aspect="auto")
        best = M.argmax(axis=0)
        for i in range(len(rows)):
            for c in range(4):
                dark = M[i, c] > 76
                ax.text(c, i, f"{M[i, c]:.1f}", ha="center", va="center", fontsize=6.0,
                        color="white" if dark else INK, fontweight="bold" if best[c] == i else "normal")
                if best[c] == i:
                    ax.add_patch(Rectangle((c - 0.46, i - 0.44), 0.92, 0.88, fill=False, ec="#e69f00", lw=0.9))
        ax.set_xticks(range(4), [G(k) for k in (1, 2, 3, 4)])
        ax.set_yticks(range(len(rows)), [short[a] for a in rows] if j == 0 else [])
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title, fontsize=6.5)
        span = M.max(axis=0) - M.min(axis=0)
        ax.set_xlabel("Column range: " + " / ".join(f"{v:.2f}" for v in span), fontsize=5.6, color=INK2)
        if j == 0:
            plabel(ax, "b", dx=-44)
        VALUES[f"fig3b_{key}"] = {a: [round(v, 2) for v in M[i]] for i, a in enumerate(rows)}
        VALUES[f"fig3b_{key}_column_range"] = [round(v, 2) for v in span]
        VALUES[f"fig3b_{key}_row_range"] = {a: round(float(M[i].max() - M[i].min()), 2) for i, a in enumerate(rows)}

    # c: calibrated threshold of every arm (rows) against every reference (columns)
    arms_t = ["tgtA", "tgtB", "tgtC", "tgtD", "tgtE", "archR18", "archCNX", "archPVT", "archPU1e3",
              "tgtBshift", "bg_maj", "tgtBshift_bg"]
    lab_t = ["A", "B", "C", "D", "E", "ResNet-18", "ConvNeXt-T", "PVT v2-b1", "Plain U-Net", "+ transl.", "+ backgr.", "+ both"]
    ax = fig.add_subplot(gs[:, 2]); ax.set_gid("c")
    Th = np.array([[T[a][str(k)]["thr"] for k in (1, 2, 3, 4)] for a in arms_t])
    L = np.log10(Th / (1 - Th))
    ax.imshow(L, cmap=THR_CMAP, vmin=-3, vmax=3, aspect="auto")
    for i in range(len(arms_t)):
        for c in range(4):
            t = Th[i, c]
            ax.text(c, i, f"{t:g}".replace("0.", "."), ha="center", va="center", fontsize=5.6,
                    color="white" if L[i, c] > 1.6 else INK)
    for yb in (4.5, 8.5):
        ax.axhline(yb, color="white", lw=1.6)
    ax.set_xticks(range(4), [G(k) for k in (1, 2, 3, 4)])
    ax.yaxis.tick_right()
    ax.set_yticks(range(len(arms_t)), lab_t, fontsize=5.8)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Calibrated threshold", fontsize=6.5)
    plabel(ax, "c", dx=-10)
    VALUES["fig3c_thresholds"] = {a: list(map(float, Th[i])) for i, a in enumerate(arms_t)}

    # d: per-case evaluation-target effect on the held-out test set
    ax = fig.add_subplot(gs[1, 0]); ax.set_gid("d")
    items = [("A, G₁ − G₄\nfixed 0.5", "tgtA", 1, 4, "dice_05"), ("A, G₁ − G₄\ncalibrated", "tgtA", 1, 4, "dice_opt"),
             ("B, G₂ − G₃\nfixed 0.5", "tgtB", 2, 3, "dice_05"), ("B, G₂ − G₃\ncalibrated", "tgtB", 2, 3, "dice_opt")]
    tick3 = ["Arm A\nG₁ − G₄\nfixed 0.5", "Arm A\nG₁ − G₄\ncalibrated", "Arm B\nG₂ − G₃\nfixed 0.5", "Arm B\nG₂ − G₃\ncalibrated"]
    rng = np.random.default_rng(3)
    out = {}
    for i, (lab, a, k1, k2, key) in enumerate(items):
        t1 = 0.5 if key == "dice_05" else Tt[a][str(k1)]["thr"]
        t2 = 0.5 if key == "dice_05" else Tt[a][str(k2)]["thr"]
        x1, x2 = pc_test(a, k1, t1), pc_test(a, k2, t2)
        x1, x2 = x1.align(x2, join="inner")
        dd = 100 * (x1 - x2).values
        jit = rng.uniform(-0.17, 0.17, len(dd))
        ax.scatter(np.full(len(dd), i - 0.08) + jit, dd, s=2.2, color=ARM_C[a], alpha=0.6, lw=0, zorder=3, rasterized=True)
        q1, med, q3 = np.percentile(dd, [25, 50, 75])
        ax.add_patch(Rectangle((i + 0.17, q1), 0.13, q3 - q1, fc="white", ec=INK, lw=0.6, zorder=4))
        ax.plot([i + 0.17, i + 0.30], [med, med], color=INK, lw=0.9, zorder=5)
        ax.plot([i + 0.235], [dd.mean()], "D", ms=2.6, color=INK, zorder=6)
        pos = 100 * (dd > 0).mean()
        ax.text(i, 106, f"{pos:.0f}% > 0", ha="center", va="bottom", fontsize=5.6, color=INK2)
        out[lab.replace("\n", " ").replace("₁", "$_1$").replace("₂", "$_2$").replace("₃", "$_3$").replace("₄", "$_4$")] = {
            "n": int(len(dd)), "mean": round(dd.mean(), 2), "median": round(med, 2), "pct_positive": round(pos, 1)}
    ax.axhline(0, color=INK2, lw=0.6, zorder=2)
    ax.set_xticks(range(len(items)), tick3, fontsize=5.6)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.5, len(items) - 0.5)
    ax.set_ylim(-45, 118)
    ax.set_ylabel("Per-case Dice change (points)")
    ax.set_title("Held-out test, predictions unchanged", fontsize=6.5)
    plabel(ax, "d", dx=-44)
    VALUES["fig3d_per_case"] = out

    # e: predicted area vs threshold
    ex = fig.add_subplot(gs[1, 1]); ex.set_gid("e")
    for k in (1, 2, 3, 4):
        val = R["gt_area_case_mean_px"][f"G{k}"]
        ex.axhline(val, color=INK3, lw=0.6, ls=(0, (3, 1.5)), zorder=2)
        if k <= 2:
            ex.text(0.9985, val + 4, f"{G(k)} {val:.0f} px", ha="right", va="bottom", fontsize=5.6, color=INK2)
        else:
            ex.text(0.00085, val + 4, f"{G(k)} {val:.0f} px", ha="left", va="bottom", fontsize=5.6, color=INK2)
    for a in arms:
        s = curves_src[curves_src.arm == a].sort_values("threshold")
        ex.plot(s.threshold, s.area_case_mean, color=ARM_C[a], lw=1.1, zorder=3)
        t = float(T5[a]["threshold"])
        v = float(s.loc[np.isclose(s.threshold, t), "area_case_mean"].iloc[0])
        ex.plot([t], [v], "o", ms=3.2, color=ARM_C[a], mec="white", mew=0.6, zorder=5)
    logit_axis(ex)
    ex.set_ylim(0, 310)
    ex.set_xlabel("Binarisation threshold (logit scale)")
    ex.set_ylabel("Mean predicted area (px)")
    ex.set_title("Predicted vs reference extent (CV)", fontsize=6.5)
    plabel(ex, "e", dx=-30)
    save(fig, "Fig3", row_groups=[{"id": "a", "panels": ["a1", "a2", "a3", "a4"]},
                                  {"id": "b", "panels": ["b1", "b2"]}, {"id": "de", "panels": ["d", "e"]}],
         column_groups=[{"id": "left", "panels": ["b1", "d"]}, {"id": "mid", "panels": ["b2", "e"]}])


# ====================================================================== Fig. 4
def fig4():
    noise = J("outputs", "analysis_full", "noise_floor.json")
    nf = noise["fixed_0_5"]["two_sigma_pts"]
    H = 4.6
    fig = plt.figure(figsize=(W, H))
    gs = fig.add_gridspec(2, 2, left=0.085, right=0.99, top=0.955, bottom=0.12, wspace=0.26, hspace=0.62,
                          width_ratios=[1.0, 1.15])
    ax = fig.add_subplot(gs[:, 0]); ax.set_gid("a")
    SOFT = ARM_C["tgtC"]
    contrasts = [("tgtC", "tgtA", "C − A  conventional comparison", INK),
                 ("tgtB", "tgtA", "B − A  mask extent only", INK3),
                 ("tgtC", "tgtB", "C − B  soft supervision only", SOFT),
                 ("tgtD", "tgtB", "D − B  soft target only vs hard", SOFT),
                 ("tgtC", "tgtE", "C − E  rater votes vs SVLS", SOFT)]
    y = 0.0
    yt, yl, out = [], [], {}
    for a, b, title, col in contrasts:
        ax.text(-5.3, y + 0.12, title, fontsize=6.0, fontweight="bold", va="center", ha="left", color=col if col != INK3 else INK2)
        y -= 0.72
        for k in (1, 2, 3, 4):
            for test, dy, mk in ((False, 0.13, "o"), (True, -0.13, "D")):
                d = _paired(a, b, k, test)
                lo, hi = boot_mean_ci(d, B=5000)
                m = d.mean()
                ax.plot([lo, hi], [y + dy] * 2, color=col, lw=1.1 if not test else 0.8, solid_capstyle="round", zorder=4)
                ax.plot([m], [y + dy], mk, ms=3.6 if not test else 3.0, color=col if not test else "white",
                        mec=col, mew=0.8, zorder=5)
                out[f"{a}-{b}_G{k}_{'test' if test else 'cv'}"] = {"n": int(len(d)), "diff": round(m, 2),
                                                                    "ci": [round(lo, 2), round(hi, 2)]}
            yt.append(y); yl.append(G(k))
            y -= 0.55
        y -= 0.22
    ax.axvspan(-nf, nf, color=BAND, zorder=0, lw=0)
    ax.axvline(0, color=INK2, lw=0.6, zorder=2)
    ax.set_yticks(yt, yl, fontsize=6.0)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-5.4, 2.4)
    ax.set_ylim(y + 0.3, 0.45)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Paired Dice difference (points), case level")
    ax.legend(handles=[Line2D([], [], marker="o", ls="-", color=INK2, mec=INK2, ms=3.4, lw=1.0, label="Cross-validation (n ≤ 634)"),
                       Line2D([], [], marker="D", ls="-", color=INK2, mfc="white", mec=INK2, ms=3.0, lw=0.8, label="Held-out test (n ≤ 158)"),
                       Patch(color=BAND, label=f"Noise floor ±{nf:.2f}")],
              loc="upper center", bbox_to_anchor=(0.45, -0.07), ncol=2, fontsize=5.8, handlelength=1.6, columnspacing=1.0)
    plabel(ax, "a", dx=-20)
    VALUES["fig4a"] = out

    bx = fig.add_subplot(gs[0, 1]); bx.set_gid("b")
    data = [_paired("tgtC", "tgtB", k, False) for k in (1, 2, 3, 4)]
    parts = bx.violinplot(data, positions=range(4), widths=0.8, showextrema=False, bw_method=0.25)
    for pc_ in parts["bodies"]:
        pc_.set_facecolor(SOFT); pc_.set_alpha(0.20); pc_.set_edgecolor("none")
    rng = np.random.default_rng(1)
    vb = {}
    tick_lab = []
    for i, d in enumerate(data):
        bx.scatter(i + rng.uniform(-0.09, 0.09, len(d)), d, s=0.9, color=SOFT, alpha=0.45, lw=0, rasterized=True, zorder=3)
        q1, med, q3 = np.percentile(d, [25, 50, 75])
        bx.plot([i + 0.22, i + 0.22], [q1, q3], color=INK, lw=2.0, zorder=4, solid_capstyle="butt")
        bx.plot([i + 0.16, i + 0.28], [med, med], color=INK, lw=0.9, zorder=5)
        bx.plot([i + 0.22], [d.mean()], "D", ms=2.6, color="white", mec=INK, mew=0.7, zorder=6)
        b_better = 100 * (d < 0).mean()
        tick_lab.append(f"{G(i + 1)}\nn = {len(d)}\nB better {b_better:.0f}%")
        vb[f"G{i+1}"] = {"n": int(len(d)), "mean": round(d.mean(), 2), "median": round(med, 2), "pct_B_better": round(b_better, 1),
                         "min": round(float(d.min()), 2), "max": round(float(d.max()), 2)}
    bx.axhline(0, color=INK2, lw=0.6, zorder=2)
    bx.set_ylim(-40, 37)
    bx.set_xticks(range(4), tick_lab, fontsize=5.6)
    bx.tick_params(axis="x", length=0)
    bx.set_ylabel("C − B per case (points)")
    bx.set_title("Soft supervision, case by case (cross-validation; all cases shown)", fontsize=6.3)
    plabel(bx, "b", dx=-26)
    VALUES["fig4b"] = vb

    gsr = gs[1, 1].subgridspec(1, 2, wspace=0.62, width_ratios=[1, 1.1])
    cx = fig.add_subplot(gsr[0, 0]); cx.set_gid("c")
    rows = [("G₁ CV", 1, False), ("G₂ CV", 2, False), ("G₁ test", 1, True), ("G₂ test", 2, True)]
    for b_, a_ in ((1.5, 0.35), (nf, 0.6), (1.0, 1.0)):
        cx.axvspan(-b_, b_, color=BAND if a_ == 1.0 else "#f5f5f5" if a_ == 0.35 else "#efefef", lw=0, zorder=0)
    for b_ in (1.0, nf, 1.5):
        for s_ in (-1, 1):
            cx.axvline(s_ * b_, color="#bdbdbd", lw=0.4, zorder=1)
    tv = {}
    for i, (lab, k, test) in enumerate(rows):
        d = _paired("tgtC", "tgtB", k, test)
        m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
        t = stats.t.ppf(0.95, len(d) - 1)
        lo, hi = m - t * se, m + t * se
        yy = len(rows) - 1 - i
        cx.plot([lo, hi], [yy, yy], color=SOFT, lw=1.3, zorder=4, solid_capstyle="round")
        cx.plot([m], [yy], "o" if not test else "D", ms=3.4, color=SOFT if not test else "white", mec=SOFT, mew=0.8, zorder=5)
        p_tost = max(stats.t.sf((m + 1.0) / se, len(d) - 1), stats.t.cdf((m - 1.0) / se, len(d) - 1))
        tv[lab.replace("₁", "$_1$").replace("₂", "$_2$")] = {"mean": round(m, 3), "ci90": [round(lo, 2), round(hi, 2)],
                                                             "p_tost_1.0": float(p_tost)}
    cx.axvline(0, color=INK2, lw=0.6, zorder=2)
    cx.set_yticks(range(len(rows)), [r[0] for r in rows][::-1], fontsize=5.8)
    cx.tick_params(axis="y", length=0)
    cx.set_xlim(-1.7, 1.7)
    cx.set_ylim(-0.6, len(rows) - 0.4)
    cx.set_xticks([-1.5, -1.0, 0, 1.0, 1.5], ["−1.5", "−1", "0", "1", "1.5"])
    cx.set_xlabel("C − B, 90% CI (points)")
    cx.set_title("Equivalence (TOST)", fontsize=6.3)
    cx.spines["left"].set_visible(False)
    plabel(cx, "c", dx=-30)
    VALUES["fig4c"] = tv

    dx = fig.add_subplot(gsr[0, 1]); dx.set_gid("d")
    seeds = json.load(open(os.path.join(SD, "noise_seeds.json")))
    per = noise["fixed_0_5"]["per_target_two_sigma"]
    dx.axhspan(-nf, nf, color=BAND, zorder=0, lw=0)
    for k in (1, 2, 3, 4):
        s2 = per[f"V>={k}"]
        dx.plot([k - 1.38, k - 0.62], [s2, s2], color=INK, lw=0.7, zorder=2)
        dx.plot([k - 1.38, k - 0.62], [-s2, -s2], color=INK, lw=0.7, zorder=2)
    for g_ in seeds:
        v = np.array(g_["v"]); dev = v - v.mean()
        x0 = g_["k"] - 1 + (-0.18 if g_["arm"] == "tgtA" else 0.18) + (-0.06 if g_["fold"] == "val_fold0" else 0.06)
        dx.scatter([x0] * 3, dev, s=6, color=ARM_C[g_["arm"]], marker="o" if g_["fold"] == "val_fold0" else "^",
                   edgecolor="white", linewidth=0.3, zorder=4)
    dx.axhline(0, color=INK2, lw=0.5, zorder=1)
    dx.set_xticks(range(4), [G(k) for k in (1, 2, 3, 4)])
    dx.tick_params(axis="x", length=0)
    dx.set_ylim(-1.6, 1.6)
    dx.set_ylabel("Seed deviation (points)")
    dx.set_title("Retraining noise (3 seeds)", fontsize=6.3)
    dx.legend(handles=[Line2D([], [], marker="o", ls="none", color=ARM_C["tgtA"], ms=3, label="Arm A"),
                       Line2D([], [], marker="o", ls="none", color=ARM_C["tgtB"], ms=3, label="Arm B"),
                       Line2D([], [], marker="o", ls="none", color=INK3, ms=3, label="Fold 0"),
                       Line2D([], [], marker="^", ls="none", color=INK3, ms=3, label="Fold 1"),
                       Line2D([], [], color=INK, lw=0.7, label="±2σ per reference")],
              loc="upper center", bbox_to_anchor=(0.3, -0.13), ncol=3, fontsize=5.4, handlelength=1.0,
              columnspacing=0.5, handletextpad=0.3)
    plabel(dx, "d", dx=-30)
    VALUES["fig4d_noise"] = {"pooled_2sigma": nf, "per_target": per}
    save(fig, "Fig4", row_groups=[{"id": "cd", "panels": ["c", "d"]}], column_groups=[],
         exemptions=[])


# ====================================================================== Fig. 5
def _per_nodule_5fold(d, arm, offs):
    fr = []
    for f in range(5):
        fr.append(pd.DataFrame(d["runs"][f"{arm}_fold{f}"]["per_nodule"]))
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
    H = 4.6
    fig = plt.figure(figsize=(W, H))
    gs = fig.add_gridspec(2, 2, left=0.085, right=0.855, top=0.955, bottom=0.10, wspace=0.52, hspace=0.30,
                          width_ratios=[1.55, 1.0], height_ratios=[0.78, 1.0])
    # a: displaced windows
    gsa = gs[0, 0].subgridspec(1, 5, wspace=0.06)
    img, vote = ex["img"].astype(float), ex["vote"]
    cy, cx_ = int(round(float(ex["cy"]))), int(round(float(ex["cx"])))
    for j, dd in enumerate([0, 8, 16, 32, 56]):
        ax = fig.add_subplot(gsa[0, j]); ax.set_gid(f"a{j + 1}")
        y0, x0 = cy - 64, cx_ - 64 + dd
        crop = img[y0:y0 + 128, x0:x0 + 128]; v = vote[y0:y0 + 128, x0:x0 + 128]
        ax.imshow(crop, cmap="gray", vmin=0, vmax=255, interpolation="bilinear")
        ax.contour((v >= 2).astype(float), [0.5], colors=[REF_C[1]], linewidths=0.9)
        ax.plot([64], [64], "+", color="white", ms=5, mew=0.7)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color(INK3); sp.set_linewidth(0.5)
        ax.set_title(f"d = {dd} px", fontsize=6.3, pad=2)
        if j == 0:
            plabel(ax, "a", dx=-14)
    # b: Dice vs offset
    bx = fig.add_subplot(gs[1, 0]); bx.set_gid("b")
    bx.fill_between([0, 32], 0, 1.0, color=BAND, zorder=0, lw=0)
    others = [r for r in da["runs"] if r not in ("tgtB_fold0", "tgtBshift_fold0")]
    for r in others:
        yv = [da["runs"][r]["mean_by_offset"][str(o)] for o in offs]
        bx.plot(offs, yv, color="#c4c4c4", lw=0.7, zorder=2)
    for df_, arm, lab in ((B, "tgtB", "B majority, centre-cropped (5-fold ensemble)"),
                          (S, "tgtBshift", "B + translation ±32 px (5-fold ensemble)")):
        m = [df_[f"d{o}"].mean() for o in offs]
        ci = np.array([boot_mean_ci(df_[f"d{o}"].values, B=5000) for o in offs])
        bx.fill_between(offs, ci[:, 0], ci[:, 1], color=ARM_C[arm], alpha=0.16, lw=0, zorder=3)
        bx.plot(offs, m, color=ARM_C[arm], lw=1.6, marker=ARM_M[arm], ms=3.4, mec="white", mew=0.5, zorder=5, label=lab)
    bx.plot([], [], color="#c4c4c4", lw=0.7, label="Nine further arms without translation (fold 0)")
    bx.text(16, 0.035, "Augmentation range", ha="center", va="bottom", fontsize=5.8, color=INK2)
    bx.set_xticks(offs)
    bx.set_xlim(-1.5, 57.5); bx.set_ylim(0, 1.17)
    bx.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); bx.spines["left"].set_bounds(0, 1.0)
    bx.set_xlabel("Offset of the evaluation window from the lesion centroid, d (px)")
    bx.set_ylabel("Dice against G₂ (held-out test, 76 nodules)")
    bx.legend(loc="upper right", bbox_to_anchor=(1.0, 1.02), fontsize=5.6, handlelength=1.6)
    plabel(bx, "b", dx=-30)
    VALUES["fig5b"] = {"B_5fold": [round(B[f"d{o}"].mean(), 4) for o in offs],
                       "Bshift_5fold": [round(S[f"d{o}"].mean(), 4) for o in offs]}
    # c: heat strip, eleven arms
    cxax = fig.add_subplot(gs[0, 1]); cxax.set_gid("c")
    runs = da["runs"]
    names = {"tgtA_fold0": "A union", "tgtB_fold0": "B majority", "tgtC_fold0": "C maj. + soft", "tgtD_fold0": "D soft only",
             "tgtE_fold0": "E SVLS", "tgtF_fold0": "F SoftSeg", "tgtBshift_fold0": "B + translation",
             "archR18_fold0": "ResNet-18", "archCNX_fold0": "ConvNeXt-T", "archPVT_fold0": "PVT v2-b1",
             "archPU1e3_fold0": "Plain U-Net"}
    order = sorted(runs, key=lambda r: runs[r]["mean_by_offset"]["0"] - runs[r]["mean_by_offset"]["8"])
    M = np.array([[runs[r]["mean_by_offset"][str(o)] for o in offs] for r in order])
    im = cxax.imshow(M, cmap=DICE_CMAP, vmin=0, vmax=0.9, aspect="auto")
    cxax.set_yticks(range(len(order)), [names[r] for r in order], fontsize=5.6)
    for tl, r in zip(cxax.get_yticklabels(), order):
        if "shift" in r:
            tl.set_fontweight("bold")
    cxax.set_xticks(range(len(offs)), offs, fontsize=5.6)
    cxax.set_xlabel("d (px)")
    cxax.tick_params(length=0)
    for sp in cxax.spines.values():
        sp.set_visible(False)
    tr = blended_transform_factory(cxax.transAxes, cxax.transData)
    for i, r in enumerate(order):
        drop = 100 * (runs[r]["mean_by_offset"]["0"] - runs[r]["mean_by_offset"]["8"])
        cxax.text(1.07, i, sgn(-drop, 1), va="center", ha="left", fontsize=5.6, transform=tr,
                  fontweight="bold" if "shift" in r else "normal")
    cxax.text(1.07, -0.75, "Δ at d = 8", va="bottom", ha="left", fontsize=5.6, color=INK2, transform=tr)
    cax = fig.add_axes([0.935, 0.70, 0.012, 0.22])
    cb = fig.colorbar(im, cax=cax, ticks=[0, 0.3, 0.6, 0.9])
    cb.ax.tick_params(labelsize=5.4, length=1.5); cb.outline.set_visible(False)
    cb.set_label("Dice (fold 0)", fontsize=5.6)
    plabel(cxax, "c", dx=-58)
    VALUES["fig5c_drop8"] = {names[r]: round(100 * (runs[r]["mean_by_offset"]["0"] - runs[r]["mean_by_offset"]["8"]), 2) for r in order}
    VALUES["fig5c_median_drop8_10arms"] = P["ten_arms_fold0"]["median_drop8_pts"]
    # d: size dependence of the position prior
    dx = fig.add_subplot(gs[1, 1]); dx.set_gid("d")
    dropB = 100 * (B.d0 - B.d8); dropS = 100 * (S.d0 - S.d8)
    rB = spearmanr(B.diameter_mm, dropB); rS = spearmanr(S.diameter_mm, dropS)
    dx.axhline(0, color=INK2, lw=0.5, zorder=1)
    dx.scatter(B.diameter_mm, dropB, s=7, color=ARM_C["tgtB"], edgecolor="white", lw=0.3, zorder=4,
               label=f"B majority: ρ = {num(rB.correlation)}, p = {sci(rB.pvalue)}")
    dx.scatter(S.diameter_mm, dropS, s=7, facecolor="white", edgecolor=ARM_C["tgtBshift"], lw=0.5, marker="s", zorder=3,
               label=f"B + translation: ρ = {num(rS.correlation)}, p = {rS.pvalue:.2f}")
    dx.set_xscale("log"); dx.set_xticks([3, 5, 10, 20], ["3", "5", "10", "20"]); dx.minorticks_off()
    dx.set_ylim(-12, 125)
    dx.set_yticks([0, 25, 50, 75, 100])
    dx.set_xlabel("Nodule equivalent diameter (mm)")
    dx.set_ylabel("Dice lost at d = 8 (points)")
    dx.legend(loc="upper right", bbox_to_anchor=(1.03, 1.02), fontsize=5.4, handletextpad=0.2, markerscale=1.2)
    plabel(dx, "d", dx=-30)
    bins = [3, 6, 10, 30]
    VALUES["fig5d"] = {"spearman_B": [round(rB.correlation, 3), float(rB.pvalue)],
                       "spearman_shift": [round(rS.correlation, 3), float(rS.pvalue)],
                       "mean_drop_by_diam": {str(k): round(v, 2) for k, v in dropB.groupby(pd.cut(B.diameter_mm, bins), observed=False).mean().items()},
                       "n_by_diam": {str(k): int(v) for k, v in pd.cut(B.diameter_mm, bins).value_counts().sort_index().items()}}
    save(fig, "Fig5", row_groups=[{"id": "a", "panels": [f"a{j}" for j in range(1, 6)]}, {"id": "bd", "panels": ["b", "d"]}],
         column_groups=[], exclude=[cax])


# ====================================================================== Fig. 6
SCOPES = ["patch128", "win128", "win192", "win256", "win384", "win512", "slice", "volume"]
SC_L = ["patch\n128", "win\n128", "win\n192", "win\n256", "win\n384", "win\n512", "full\nslice", "whole\nvolume"]
ARMS6 = [("tgtB", "B majority (centre-cropped)"), ("bg_maj", "B + background sampling"),
         ("tgtBshift", "B + translation augmentation"), ("tgtBshift_bg", "B + both (merged arm)")]


def fig6():
    S = J("outputs", "analysis_scope", "scope_analysis_merged2.json")
    T = thr_cv()
    sc = scope_case()
    ex = np.load(os.path.join(SD, "example_slice.npz"))
    H = 4.4
    fig = plt.figure(figsize=(W, H))
    gs = fig.add_gridspec(2, 2, left=0.085, right=0.99, top=0.93, bottom=0.10, wspace=0.26, hspace=0.50,
                          width_ratios=[1.0, 1.0], height_ratios=[1.12, 1.0])
    ax = fig.add_axes([0.0, 0.50, 0.30, 0.42]); ax.set_gid("a")
    img = ex["img"].astype(float); vote = ex["vote"]
    cy, cx = float(ex["cy"]), float(ex["cx"])
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="bilinear")
    ax.contour((vote >= 2).astype(float), [0.5], colors=[REF_C[1]], linewidths=0.7)
    ramp = ["#ffffff", "#d9d9d9", "#b3b3b3", "#8c8c8c", "#666666"]
    for w_, c_ in zip((128, 192, 256, 384, 512), ramp):
        x0 = np.clip(cx - w_ / 2, 0, 512 - w_); y0 = np.clip(cy - w_ / 2, 0, 512 - w_)
        ax.add_patch(Rectangle((x0, y0), w_ - 1, w_ - 1, fill=False, ec=c_, lw=0.9 if w_ < 512 else 1.6))
        ax.text(x0 + 6, y0 + 6, f"{w_}", color="white", fontsize=5.6, va="top", ha="left", path_effects=halo(1.4, "#000000"))
    ax.set_xlim(0, 511); ax.set_ylim(511, 0)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Full slice, one forward pass;\nonly the scoring window changes", fontsize=6.2)
    flabel(fig, "a", 0.0, 0.995)

    bx = fig.add_axes([0.40, 0.545, 0.585, 0.385]); bx.set_gid("b")
    x = np.arange(len(SCOPES))
    bx.fill_between([0.5, 5.5], 0, 1.0, color=BAND, zorder=0, lw=0)
    bx.text(3.0, 0.985, "Field-of-view ladder: one probability map, five windows", ha="center", va="top", fontsize=5.8, color=INK2)
    lad = {}
    for a, lab in ARMS6:
        t = T[a]["2"]["thr"]
        s = sc[(sc.arm == a) & np.isclose(sc.threshold, t)]
        m, lo, hi = [], [], []
        for scp in SCOPES:
            v = s[s.scope == scp].dice.values
            m.append(v.mean()); l_, h_ = boot_mean_ci(v, B=5000); lo.append(l_); hi.append(h_)
        assert abs(m[1] - S["table1_ladder"][f"{a}_v2"]["win128"]) < 1e-3, (a, m[1])
        bx.fill_between(x, lo, hi, color=ARM_C[a], alpha=0.16, lw=0, zorder=2)
        bx.plot(x, m, color=ARM_C[a], lw=1.6 if a == "tgtBshift_bg" else 1.2, marker=ARM_M[a], ms=3.2, mec="white", mew=0.5,
                zorder=4, label=lab)
        lad[a] = [round(v, 4) for v in m]
    bx.set_xticks(x, SC_L, fontsize=5.8)
    bx.tick_params(axis="x", length=0)
    bx.set_xlim(-0.4, len(SCOPES) - 0.6); bx.set_ylim(0, 1.2)
    bx.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); bx.spines["left"].set_bounds(0, 1.0)
    bx.set_ylabel("Dice against G₂\n(held-out test, n = 158)")
    bx.legend(loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=2, fontsize=5.8, handlelength=1.6, columnspacing=1.5)
    plabel(bx, "b", dx=-34)
    VALUES["fig6b_ladder"] = lad

    cxx = fig.add_subplot(gs[1, 0]); cxx.set_gid("c")
    rng = np.random.default_rng(5)
    loss_out = {}
    for i, (a, lab) in enumerate(ARMS6):
        t = T[a]["2"]["thr"]
        s = sc[(sc.arm == a) & np.isclose(sc.threshold, t)].pivot(index="case_id", columns="scope", values="dice")
        d = 100 * (s["win128"] - s["win512"]).dropna().values
        cxx.scatter(rng.uniform(-0.18, 0.18, len(d)) + i - 0.06, d, s=2.0, color=ARM_C[a], alpha=0.55, lw=0, zorder=3, rasterized=True)
        q1, med, q3 = np.percentile(d, [25, 50, 75])
        cxx.plot([i + 0.25, i + 0.25], [q1, q3], color=INK, lw=2.0, solid_capstyle="butt", zorder=4)
        cxx.plot([i + 0.20, i + 0.30], [med, med], color=INK, lw=0.8, zorder=5)
        r = [q for q in S["table4_paired"] if q["arm"] == a and q["level"] == 2 and q["segment"] == "纯视野效应"][0]
        assert abs(-r["diff"] - d.mean()) < 0.01, (a, r["diff"], d.mean())
        lo, hi = -r["ci"][1], -r["ci"][0]
        cxx.text(i, 114, f"{num(d.mean())}\n[{num(lo)}, {num(hi)}]", ha="center", va="top", fontsize=5.6, color=INK, linespacing=1.3)
        loss_out[a] = {"n": int(len(d)), "mean": round(d.mean(), 2), "ci": [round(lo, 2), round(hi, 2)], "median": round(med, 2)}
    cxx.axhline(0, color=INK2, lw=0.5, zorder=2)
    cxx.set_xticks(range(4), ["B", "+ background", "+ translation", "+ both"], fontsize=5.8)
    cxx.tick_params(axis="x", length=0)
    cxx.set_ylim(-35, 116)
    cxx.set_yticks([-25, 0, 25, 50, 75])
    cxx.set_ylabel("Dice lost, window 128 → 512\nper case (points)")
    plabel(cxx, "c", dx=-34)
    VALUES["fig6c_loss"] = loss_out

    dx = fig.add_subplot(gs[1, 1]); dx.set_gid("d")
    t3 = S["table3_per_scope_opt"]
    lad_sc = SCOPES[1:6]
    xx = np.arange(len(lad_sc))
    drift = {}
    for row, (a, lab) in enumerate(ARMS6[2:]):
        fixed = [S["table1_ladder"][f"{a}_v2"][s_] for s_ in lad_sc]
        opt = [t3[f"{a}_v2_{s_}"]["dice_opt"] for s_ in lad_sc]
        thr = [t3[f"{a}_v2_{s_}"]["thr"] for s_ in lad_sc]
        col = ARM_C[a]
        dx.plot(xx, fixed, color=col, lw=1.2, marker=ARM_M[a], ms=3.0, mec="white", mew=0.4, zorder=4)
        dx.plot(xx, opt, color=col, lw=1.0, ls=(0, (3, 1.5)), marker=ARM_M[a], ms=3.0, mfc="white", mec=col, mew=0.6, zorder=4)
        yrow = 0.935 - row * 0.062
        dx.text(-0.62, yrow, "transl. t*" if a == "tgtBshift" else "both t*", ha="right", va="center", fontsize=5.4, color=col)
        for xi, t in enumerate(thr):
            dx.text(xi, yrow, f"{t:g}".replace("0.", "."), ha="center", va="center", fontsize=5.4, color=col)
        drift[a] = {"thr_per_scope": thr, "loss_fixed_pts": round(100 * (fixed[0] - fixed[-1]), 2),
                    "loss_recalibrated_pts": round(100 * (opt[0] - opt[-1]), 2)}
    dx.legend(handles=[Line2D([], [], color=INK2, lw=1.2, label="Validation-calibrated threshold"),
                       Line2D([], [], color=INK2, lw=1.0, ls=(0, (3, 1.5)), label="Re-optimised per window (t*)")],
              loc="lower left", fontsize=5.6, handlelength=1.8)
    dx.set_xticks(xx, ["128", "192", "256", "384", "512"])
    dx.set_xlim(-0.5, 4.6)
    dx.set_xlabel("Evaluation window (px)")
    dx.set_ylabel("Dice against G₂")
    dx.set_ylim(0.1, 0.98)
    dx.set_yticks([0.2, 0.4, 0.6, 0.8])
    plabel(dx, "d", dx=-30)
    VALUES["fig6d_drift"] = drift
    save(fig, "Fig6", row_groups=[{"id": "cd", "panels": ["c", "d"]}], column_groups=[])


# ====================================================================== Fig. 7
def effect_rows():
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
        ("Evaluation target V ≥ 1 vs V ≥ 4, arm A, threshold 0.5", E["evaluation_target_span_pts"]["point"],
         E["evaluation_target_span_pts"]["ci95"], cv_tgt, "protocol", None),
        ("Position offset of 8 px (median of ten arms)", O["median_drop8_pts"], None, None, "data",
         sorted(v["drop8_pts"] for v in O["arms"].values())),
        ("Field of view 128→512 px, both confounds removed", -E["fov_effect_pts"]["point"],
         sorted(-np.array(E["fov_effect_pts"]["ci95"])), None, "protocol", None),
        ("Binarisation threshold, arm A at V ≥ 4", E["threshold_effect_pts"]["point"], E["threshold_effect_pts"]["ci95"],
         cv_thr, "protocol", None),
        ("Cost of translation augmentation (centred test)", E["translation_cost_pts"]["point"],
         E["translation_cost_pts"]["ci95"], cv_shift, "data", None),
        ("Supervision target, five arms at V ≥ 4", E["supervision_range_pts"]["point"], E["supervision_range_pts"]["ci95"],
         cv_sup, "method", None),
        ("Architecture, five models at V ≥ 2", E["architecture_range_pts"]["point"], E["architecture_range_pts"]["ci95"],
         cv_arch, "method", None),
    ]
    rows.sort(key=lambda r: -r[1])
    nf = N["fixed_0_5"]["two_sigma_pts"]
    rows.append(("Retraining with a new seed (2σ, validation)", nf, [0.96, 1.58], nf, "noise", None))
    return rows, nf


def fig7():
    rows, nf = effect_rows()
    H = 4.3
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0.385, 0.385, 0.385, 0.575]); ax.set_gid("a")
    y = np.arange(len(rows))[::-1]
    ax.axvspan(0, nf, color=BAND, zorder=0, lw=0)
    out = []
    for yi, (lab, v, ci, cvv, cat, dots) in zip(y, rows):
        open_bar = lab.endswith("position-invariant arm")
        if cat == "noise":
            ax.barh(yi, v, height=0.56, color="#e3e3e3", edgecolor=INK3, lw=0.6, zorder=3)
        elif open_bar:
            ax.barh(yi, v, height=0.56, color="white", edgecolor=CAT_C[cat], lw=0.9, zorder=3)
        else:
            ax.barh(yi, v, height=0.56, color=CAT_C[cat], edgecolor=INK2 if cat == "method" else CAT_C[cat], lw=0.6, zorder=3)
        right = v
        if ci is not None:
            ax.plot(ci, [yi, yi], color=INK if cat != "protocol" or open_bar else "#7a7a7a", lw=0.7, zorder=5)
            for b in ci:
                ax.plot([b, b], [yi - 0.11, yi + 0.11], color=INK if cat != "protocol" or open_bar else "#7a7a7a", lw=0.7, zorder=5)
            right = max(right, ci[1])
        if dots:
            ax.plot(dots, [yi] * len(dots), "o", ms=2.4, color="white", mec=INK, mew=0.5, zorder=6)
            right = max(right, max(dots))
        if cvv is not None and cat != "noise":
            ax.plot([cvv], [yi], "D", ms=3.0, color="white", mec=INK if not (cat == "protocol" and not open_bar) else "white",
                    mfc="white" if not (cat == "protocol" and not open_bar) else "none", mew=0.8, zorder=7)
        ax.text(right + 0.9, yi, num(v), va="center", ha="left", fontsize=6.0, fontweight="bold", zorder=6)
        out.append({"row": lab, "test": round(v, 2), "ci": None if ci is None else [round(c, 2) for c in ci],
                    "cv": None if cvv is None else round(cvv, 2)})
    ax.set_yticks(y, [r[0] for r in rows], fontsize=6.0)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 50)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Change in Dice (points) when one factor is varied")
    ax.legend(handles=[Patch(facecolor=CAT_C["protocol"], label="Evaluation protocol"),
                       Patch(facecolor="white", edgecolor=CAT_C["protocol"], lw=0.9, label="Evaluation protocol, training confound present"),
                       Patch(facecolor=CAT_C["data"], label="Data / training protocol"),
                       Patch(facecolor=CAT_C["method"], edgecolor=INK2, lw=0.6, label="Learning design"),
                       Patch(facecolor="#e3e3e3", edgecolor=INK3, lw=0.6, label="Run-to-run noise floor"),
                       Line2D([], [], marker="D", ls="none", mfc="white", mec=INK, ms=3.2, label="Cross-validation estimate")],
              loc="lower right", bbox_to_anchor=(1.02, 0.0), fontsize=5.6, handlelength=1.0, handleheight=0.8)
    flabel(fig, "a", 0.0, 0.995)
    VALUES["fig7a"] = out

    bx = fig.add_axes([0.80, 0.385, 0.19, 0.575]); bx.set_gid("b")
    bx.axvline(1, color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=1)
    bx.axvline(10, color=INK3, lw=0.5, ls=(0, (1, 2)), zorder=1)
    for yi, (lab, v, ci, cvv, cat, dots) in zip(y, rows):
        if cat == "noise":
            continue
        open_m = lab.endswith("position-invariant arm")
        col = INK if cat == "protocol" else ("#7a7a7a" if cat == "data" else "#a6a6a6")
        if ci is not None:
            bx.plot([max(ci[0], 0.05) / nf, ci[1] / nf], [yi, yi], color=col, lw=1.0, zorder=4)
        bx.plot([v / nf], [yi], "o", ms=3.6, color="white" if open_m else col, mec=col if cat != "method" else INK2, mew=0.8, zorder=5)
    bx.set_xscale("log")
    bx.set_xlim(0.3, 60)
    bx.set_xticks([0.5, 1, 2, 5, 10, 20, 50], ["0.5", "1", "2", "5", "10", "20", "50"])
    bx.minorticks_off()
    bx.set_yticks(y[:-1], [])
    bx.tick_params(axis="y", length=0)
    bx.spines["left"].set_visible(False)
    bx.set_ylim(-0.6, len(rows) - 0.4)
    bx.set_xlabel("Effect ÷ noise floor (×)")
    flabel(fig, "b", 0.785, 0.995)

    Tt = thr_test(); T = thr_cv()
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
    cx = fig.add_axes([0.03, 0.085, 0.96, 0.13]); cx.set_gid("c")
    binsl = np.logspace(np.log10(1), np.log10(80), 60)
    JR = J("outputs", "analysis", "effect_uncertainty.json")["joint_ratio_bootstrap"]
    for r, col, pt in ((r1, "#4d4d4d", p1), (r2, "#b3b3b3", p2)):
        cx.hist(r, bins=binsl, color=col, alpha=0.85, lw=0, zorder=3)
        lo, hi = np.percentile(r, [2.5, 97.5])
        VALUES[f"fig7c_{'sup' if col == '#4d4d4d' else 'arch'}"] = {"point": round(pt, 2), "ci95": [round(lo, 2), round(hi, 2)],
                                                                     "n_common": int(len(base))}
    ymax = cx.get_ylim()[1]
    cx.set_ylim(0, ymax * 1.0)
    for col, lab, key, yy in (("#4d4d4d", "Field of view ÷ supervision-target range", "ratio_protocol_supervision", 1.30),
                              ("#b3b3b3", "Field of view ÷ architecture range", "ratio_protocol_architecture", 1.62)):
        pt = JR[key]["point"]; lo, hi = JR[key]["ci95"]
        cx.plot([lo, hi], [ymax * yy] * 2, color=col if col != "#b3b3b3" else "#8c8c8c", lw=1.0, clip_on=False)
        cx.plot([pt], [ymax * yy], "o", ms=3.0, color=col if col != "#b3b3b3" else "#8c8c8c", clip_on=False)
        cx.text(lo / 1.12, ymax * yy, f"{lab}: {pt:.2f}× [{lo:.2f}, {hi:.2f}]", fontsize=5.8, va="center", ha="right",
                color=INK, clip_on=False)
    cx.axvline(1, color=INK2, lw=0.6, ls=(0, (3, 2)))
    cx.set_xscale("log"); cx.set_xlim(0.8, 80)
    cx.set_xticks([1, 2, 5, 10, 20, 50], ["1", "2", "5", "10", "20", "50"]); cx.minorticks_off()
    cx.set_yticks([])
    cx.spines["left"].set_visible(False)
    cx.set_xlabel("Joint bootstrap ratio (×), 124 common test cases")
    flabel(fig, "c", 0.0, 0.30)
    assert abs(p1 - 7.03) < 0.05 and abs(p2 - 15.57) < 0.1, (p1, p2)
    save(fig, "Fig7", row_groups=[{"id": "ab", "panels": ["a", "b"]}], column_groups=[],
         exemptions=[{"panels": ["a", "b"], "checks": ["panel-width"], "reason": "b is a narrow companion axis to a"}])


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
    Tt = thr_test(); T = thr_cv()
    CL = J("outputs", "analysis", "case_lesion_sensitivity.json")["sets"]
    LU = json.load(open(os.path.join(SD, "lesion_unit.json")))
    H = 4.1
    fig = plt.figure(figsize=(W, H))
    gs = fig.add_gridspec(2, 2, left=0.08, right=0.99, top=0.95, bottom=0.10, wspace=0.26, hspace=0.62,
                          width_ratios=[1.45, 1.0])
    ax = fig.add_subplot(gs[0, 0]); ax.set_gid("a")
    strata = [("all", "All nodules"), ("3-5", "3–5 mm"), ("5-10", "5–10 mm"), ("10-30", "10–30 mm")]
    effects = [("Evaluation target", INK, "o"), ("Threshold", INK, "s"), ("Supervision target", "#8c8c8c", "o"),
               ("Architecture", "#8c8c8c", "s")]
    res = {}
    ax.axhspan(0, 1.19, color=BAND, zorder=0, lw=0)
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
            xp = si + (ei - 1.5) * 0.18
            ax.plot([xp, xp], ci, color=col, lw=0.9, zorder=3)
            ax.plot([xp], [pt], mk, ms=3.6, color=col if mk == "o" else "white", mec=col, mew=0.8, zorder=4)
        strata[si] = (st, f"{stl}\n({ns[0]} cases)")
    ax.set_xticks(range(4), [s_[1] for s_ in strata], fontsize=5.8)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.5, 3.5); ax.set_ylim(0, 52)
    ax.set_ylabel("Effect size (Dice points)")
    ax.set_title("Held-out test, by nodule size", fontsize=6.4)
    ax.legend(handles=[Line2D([], [], marker=m, ls="none", color=c, mfc=c if m == "o" else "white", mec=c, ms=3.6, label=n)
                       for n, c, m in effects], loc="upper right", bbox_to_anchor=(1.0, 1.02), fontsize=5.8, ncol=2,
              columnspacing=0.8, handletextpad=0.2)
    plabel(ax, "a", dx=-28)
    VALUES["fig8a_strata"] = res

    bx = fig.add_subplot(gs[0, 1]); bx.set_gid("b")
    names = ["Evaluation\ntarget", "Threshold", "Supervision\ntarget", "Architecture"]
    case_cv_ = [25.58, 9.42, 2.65, 0.96]
    case_te = [28.58, 9.99, 3.12, 1.19]
    les_cv = [CL["cv"]["protocol_effect"]["lesion"]["range_pts"], LU["cv_tgtA_v4_lesion_unit"]["threshold_effect_pts"],
              CL["cv"]["supervision_range"]["lesion"]["range_pts"], CL["cv"]["architecture_range"]["lesion"]["range_pts"]]
    les_te = [CL["test"]["protocol_effect"]["lesion"]["range_pts"], LU["test_tgtA_v4_lesion_unit"]["threshold_effect_pts"],
              CL["test"]["supervision_range"]["lesion"]["range_pts"], CL["test"]["architecture_range"]["lesion"]["range_pts"]]
    cols = [INK, INK, "#8c8c8c", "#8c8c8c"]
    bx.axhspan(0.6, 1.19, color=BAND, zorder=0, lw=0)
    for i in range(4):
        for dx_, c_, l_, mk in ((-0.14, case_cv_[i], les_cv[i], "o"), (0.14, case_te[i], les_te[i], "D")):
            bx.plot([i + dx_ - 0.05, i + dx_ + 0.05], [c_, l_], color=cols[i], lw=0.8, zorder=3)
            bx.plot([i + dx_ - 0.05], [c_], mk, ms=3.2, color=cols[i], zorder=4)
            bx.plot([i + dx_ + 0.05], [l_], mk, ms=3.2, color="white", mec=cols[i], mew=0.8, zorder=4)
    bx.set_yscale("log"); bx.set_ylim(0.6, 45)
    bx.set_yticks([1, 2, 5, 10, 20, 40], ["1", "2", "5", "10", "20", "40"]); bx.minorticks_off()
    bx.set_xticks(range(4), names, fontsize=5.8)
    bx.tick_params(axis="x", length=0)
    bx.set_xlim(-0.5, 3.5)
    bx.set_ylabel("Effect size (Dice points, log)")
    bx.set_title("Unit of analysis", fontsize=6.4)
    bx.legend(handles=[Line2D([], [], marker="o", ls="none", color=INK2, ms=3.2, label="Case, CV"),
                       Line2D([], [], marker="D", ls="none", color=INK2, ms=3.0, label="Case, test"),
                       Line2D([], [], marker="o", ls="none", mfc="white", mec=INK2, ms=3.2, label="Lesion, CV"),
                       Line2D([], [], marker="D", ls="none", mfc="white", mec=INK2, ms=3.0, label="Lesion, test")],
              loc="upper right", bbox_to_anchor=(1.02, 1.03), fontsize=5.6, ncol=2, columnspacing=0.6, handletextpad=0.2)
    plabel(bx, "b", dx=-28)
    VALUES["fig8b_units"] = {"case_cv": case_cv_, "lesion_cv": [round(v, 2) for v in les_cv],
                             "case_test": case_te, "lesion_test": [round(v, 2) for v in les_te]}

    cx = fig.add_subplot(gs[1, 0]); cx.set_gid("c")
    xs, ys = [], []
    cx.plot([55, 90], [55, 90], color=INK3, lw=0.6, ls=(0, (3, 2)), zorder=1)
    for a in ("tgtA", "tgtB", "tgtC", "tgtD", "tgtE", "archR18", "archCNX", "archPVT", "archPU1e3"):
        for k in "1234":
            for key, mk in (("dice_05", "o"), ("dice_opt", "D")):
                x_, y_ = 100 * T[a][k][key], 100 * Tt[a][k][key]
                xs.append(x_); ys.append(y_)
                cx.plot([x_], [y_], mk, ms=3.0 if mk == "o" else 2.8, mec=REF_C[int(k) - 1] if mk == "D" else "white",
                        mfc=REF_C[int(k) - 1] if mk == "o" else "white", mew=0.6 if mk == "D" else 0.3, zorder=3)
    r = np.corrcoef(xs, ys)[0, 1]; mad = np.mean(np.abs(np.array(xs) - np.array(ys)))
    cx.text(0.04, 0.96, f"72 cells, r = {r:.3f}\nmean |Δ| = {mad:.2f} points", transform=cx.transAxes, va="top", fontsize=5.8, color=INK)
    cx.set_xlim(55, 90); cx.set_ylim(55, 90)
    cx.set_xlabel("Cross-validation Dice (points)"); cx.set_ylabel("Held-out test Dice (points)")
    cx.set_title("Cross-evaluation matrix: 9 arms × 4 references × 2 operating points", fontsize=6.0)
    hk = [Line2D([], [], marker="o", ls="none", mfc=REF_C[k - 1], mec="white", ms=3.4, label=f"V ≥ {k}") for k in (1, 2, 3, 4)]
    hk += [Line2D([], [], marker="o", ls="none", mfc=INK3, mec="white", ms=3.2, label="Fixed 0.5"),
           Line2D([], [], marker="D", ls="none", mfc="white", mec=INK3, ms=3.0, label="Calibrated")]
    cx.legend(handles=hk, loc="lower right", bbox_to_anchor=(0.99, 0.0), fontsize=5.6, ncol=2, columnspacing=0.6,
              handletextpad=0.2)
    plabel(cx, "c", dx=-28)
    VALUES["fig8c"] = {"r": round(r, 4), "mean_abs_diff": round(mad, 3), "n": len(xs)}

    dx = fig.add_subplot(gs[1, 1]); dx.set_gid("d")
    pairs = [("tgtC", "tgtA"), ("tgtB", "tgtA"), ("tgtC", "tgtB"), ("tgtD", "tgtB"), ("tgtD", "tgtC"), ("tgtC", "tgtE")]
    xs, ys = [], []
    sign_agree = 0
    dx.plot([-4, 2], [-4, 2], color=INK3, lw=0.6, ls=(0, (3, 2)), zorder=1)
    dx.axhline(0, color=LIGHT, lw=0.5, zorder=0); dx.axvline(0, color=LIGHT, lw=0.5, zorder=0)
    for a, b in pairs:
        for k in (1, 2, 3, 4):
            dcv, dte = _paired(a, b, k, False), _paired(a, b, k, True)
            lo1, hi1 = boot_mean_ci(dcv, B=5000); lo2, hi2 = boot_mean_ci(dte, B=5000)
            x_, y_ = dcv.mean(), dte.mean()
            agree = np.sign(x_) == np.sign(y_)
            dx.plot([lo1, hi1], [y_, y_], color=INK3, lw=0.5, zorder=2)
            dx.plot([x_, x_], [lo2, hi2], color=INK3, lw=0.5, zorder=2)
            dx.plot([x_], [y_], "o", ms=3.0, mfc=INK if agree else "white", mec=INK, mew=0.6, zorder=3)
            xs.append(x_); ys.append(y_); sign_agree += agree
    r = np.corrcoef(xs, ys)[0, 1]
    dx.text(0.04, 0.96, f"24 contrasts\nr = {r:.2f}\nsame sign {sign_agree}/24",
            transform=dx.transAxes, va="top", fontsize=5.8, color=INK)
    dx.set_xlim(-3.9, 1.9); dx.set_ylim(-3.9, 1.9)
    dx.set_xlabel("Cross-validation contrast (points)"); dx.set_ylabel("Held-out test contrast (points)")
    dx.set_title("24 paired contrasts, with 95% CIs", fontsize=6.0)
    plabel(dx, "d", dx=-28)
    VALUES["fig8d"] = {"r": round(r, 3), "same_sign": int(sign_agree)}
    save(fig, "Fig8")


# ====================================================================== Fig. 9
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
    H = 4.6
    fig = plt.figure(figsize=(W, H))
    gsab = fig.add_gridspec(1, 2, left=0.085, right=0.985, top=0.955, bottom=0.60, wspace=0.26)
    modm = {"CT": "s", "MRI": "o"}
    OFF = [{"brain-tumour t2": (26, 0), "LIDC-IDRI": (0, 9), "brain-growth": (0, 9), "prostate t2": (0, 9),
            "brain-tumour t3": (-4, 9), "brain-tumour t1": (-8, -9), "kidney": (0, 9), "prostate t1": (-4, 10)},
           {"brain-tumour t2": (26, 0), "LIDC-IDRI": (0, 9), "brain-growth": (0, 9), "prostate t2": (12, 9),
            "brain-tumour t3": (-14, 9), "brain-tumour t1": (-6, -9), "kidney": (10, 8), "prostate t1": (-2, -9)}]
    for pi, (yv, ylab, rr) in enumerate([(pr, "Evaluation-target (protocol) effect", (rho_p8, p_p8, rho_p7, p_p7)),
                                         (su, "Supervision-target effect", (rho_s8, p_s8, rho_s7, p_s7))]):
        ax = fig.add_subplot(gsab[0, pi]); ax.set_gid("ab"[pi])
        for i, (key, name, n, mod) in enumerate(TASKS):
            mk = "*" if key == "LIDC" else modm[mod]
            ax.scatter(x[i], yv[i], s=(14 + 5 * n) * (2.6 if key == "LIDC" else 1), marker=mk,
                       color=INK if key == "LIDC" else "#8c8c8c", edgecolor="white", linewidth=0.5, zorder=4)
            ox, oy = OFF[pi][name]
            ax.annotate(name, xy=(x[i], yv[i]), xytext=(ox, oy), textcoords="offset points", ha="center", va="center",
                        fontsize=5.6, color=INK if key == "LIDC" else INK2, fontweight="bold" if key == "LIDC" else "normal")
        p7 = f"{rr[3]:.4f}" if rr[3] < 0.001 else f"{rr[3]:.3f}"
        ax.text(0.98, 0.98, f"Spearman ρ, all 8 tasks = {num(rr[0])} (exact p = {rr[1]:.3f})\n"
                f"7 QUBIQ tasks only = {num(rr[2])} (exact p = {p7})",
                transform=ax.transAxes, ha="right", va="top", fontsize=5.8, color=INK)
        ax.set_xlim(0.1, 1.02)
        ax.set_ylim(-4.5 if pi == 0 else -3.0, 36 if pi == 0 else 21.5)
        ax.set_xlabel("Rater agreement index |V ≥ N| / |V ≥ 1|")
        ax.set_ylabel(f"{ylab}\n(Dice points, calibrated)")
        plabel(ax, "ab"[pi], dx=-30)
    fig.legend(handles=[Line2D([], [], marker="s", ls="none", color="#8c8c8c", ms=4, label="QUBIQ, CT"),
                        Line2D([], [], marker="o", ls="none", color="#8c8c8c", ms=4, label="QUBIQ, MRI"),
                        Line2D([], [], marker="*", ls="none", color=INK, ms=7, label="LIDC-IDRI (this study)"),
                        Line2D([], [], marker="o", ls="none", color="#bdbdbd", ms=2.5, label="Marker size: rater count (3–7)")],
               loc="upper center", bbox_to_anchor=(0.5, 0.495), ncol=4, fontsize=5.8, columnspacing=1.2, handletextpad=0.3)
    VALUES["fig9_family"] = fam["family_aggregated_5pt"]

    order = sorted(range(len(TASKS)), key=lambda i: x[i])
    gs = fig.add_gridspec(1, 8, left=0.065, right=0.995, top=0.335, bottom=0.085, wspace=0.45)
    armc = {"A": ARM_C["tgtA"], "B": ARM_C["tgtB"], "C": ARM_C["tgtC"]}
    for j, i in enumerate(order):
        key, name, n, mod = TASKS[i]
        ax = fig.add_subplot(gs[0, j]); ax.set_gid(f"c{j + 1}")
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
        ax.set_xlim(0.05, 1.05); ax.set_xticks([1 / n, 1], ["1/N", "N/N"], fontsize=5.4)
        ax.set_title(f"{name}\nindex {x[i]:.2f}", fontsize=5.6, pad=2, fontweight="bold" if key == "LIDC" else "normal")
        ax.tick_params(axis="y", labelsize=5.4)
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
        if j == 0:
            ax.set_ylabel("Dice (points)")
            plabel(ax, "c", dx=-26, dy=16)
        VALUES.setdefault("fig9c", {})[name] = cv
    fig.text(0.53, 0.012, "Evaluation reference V ≥ k as a fraction of raters, k/N (tasks ordered by agreement index)",
             ha="center", fontsize=6.0)
    fig.legend(handles=[Line2D([], [], color=armc[a], lw=1.2, marker="o", ms=2.4, mec="white", label=l) for a, l in
                        (("A", "Union-trained (A)"), ("B", "Majority-trained (B)"), ("C", "Majority + soft (C)"))],
               loc="upper center", bbox_to_anchor=(0.5, 0.435), ncol=3, fontsize=5.8, columnspacing=1.5)
    VALUES["fig9_points"] = {TASKS[i][1]: {"agreement": round(x[i], 4), "protocol": round(pr[i], 2), "supervision": round(su[i], 2)}
                             for i in range(len(TASKS))}
    save(fig, "Fig9", row_groups=[{"id": "ab", "panels": ["a", "b"]}, {"id": "c", "panels": [f"c{j}" for j in range(1, 9)]}],
         column_groups=[])


if __name__ == "__main__":
    which = sys.argv[1:] or [f"fig{i}" for i in range(1, 10)]
    for w_ in which:
        globals()[w_]()

    def _conv(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    vfile = os.path.join(OUT, "figure_values_v24.json")
    old = json.load(open(vfile, encoding="utf-8")) if (os.path.exists(vfile) and len(which) < 9) else {}
    qa_old = old.get("_qa_alignment", {}); qa_old.update(VALUES.get("_qa_alignment", {}))
    old.update(VALUES); old["_qa_alignment"] = qa_old
    with open(vfile, "w", encoding="utf-8") as f:
        json.dump(old, f, ensure_ascii=False, indent=1, default=_conv)
    print("values ->", vfile)
