# -*- coding: utf-8 -*-
"""make_figure_source_data_v23.py — aggregate per-case source data for the v23 figures (2026-10-11).

Reads only released experiment outputs (no GPU, no retraining) and writes compact
case-level tables to outputs/figures/v23/source_data/.  Aggregation follows
scripts/analyze_symmetric.py exactly: an instance enters the k-th reference only if its
V>=k mask is non-empty; Dice is averaged within (fold, case) first; on the held-out test
set the five fold models share fold="test", so the within-case mean pools the five models
(= the five-fold ensemble record used throughout the paper).

Outputs
  instances_cv.csv / instances_test.csv  one row per nodule-slice instance (areas, diameter)
  case_cv.csv.gz     arm, fold, case_id, threshold, k, dice, n     (all 21 CV arms, 33 thresholds)
  case_test.csv.gz   arm, case_id, stratum, threshold, k, dice, n  (11 test arms; stratum = all/3-5/5-10/10-30)
  scope_case.csv.gz  arm, case_id, scope, threshold, dice          (field-of-view ladder, level 2)
  lesion_unit.json   lesion-unit (instance-level) threshold effect for unit-of-analysis sensitivity
  noise_seeds.json   per-seed case-level Dice of the noise-floor runs (arms A, B x folds 0-1 x 3 seeds x 4 refs)
  qubiq_curves.json  QUBIQ: case-level Dice against every nested reference, per arm (fixed 0.5 and per-run optimum)
  gallery_candidates.npz/.json  candidate nodule crops for Fig. 2a / Fig. 1a (indices used are listed in make_figures_v23.py)
  example_slice.npz  full slice LIDC-IDRI-0056 / slice 82 (held-out test) for Figs. 5a and 6a
Usage (repository root):  python scripts/make_figure_source_data_v23.py
"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.environ.get("FIG_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "figures", "v23", "source_data")
os.makedirs(OUT, exist_ok=True)
K = (1, 2, 3, 4)
USE = ["run", "fold", "case_id", "diameter_mm", "threshold"] + [f"dice_v{k}" for k in K] + \
      [f"gt_area_v{k}" for k in K]


def stratum(d):
    return np.where(d < 5, "3-5", np.where(d < 10, "5-10", "10-30"))


def agg_chunks(paths, keys_extra, with_stratum, chunksize=600_000):
    """Sum/count aggregation so that a 0.5 GB CSV fits in a 3 GB VM."""
    acc = {}
    for p in paths:
        print("  reading", os.path.relpath(p, ROOT), flush=True)
        for ch in pd.read_csv(p, usecols=USE, chunksize=chunksize,
                              dtype={"run": "category", "fold": "category", "case_id": "category"}):
            ch["arm"] = ch.run.astype(str).str.replace(r"_fold\d+", "", regex=True)  # seed runs: seed_tgtA_fold1_s1 -> seed_tgtA_s1
            ch["threshold"] = ch.threshold.round(4)
            ch["stratum"] = stratum(ch.diameter_mm.values) if with_stratum else "all"
            for k in K:
                s = ch[ch[f"gt_area_v{k}"] > 0]
                groups = [["arm"] + keys_extra + ["case_id", "threshold"]]
                if with_stratum:
                    groups = [g + ["stratum"] for g in groups]
                for g in groups:
                    r = s.groupby(g, observed=True)[f"dice_v{k}"].agg(["sum", "count"])
                    key = (k, tuple(g))
                    acc[key] = r if key not in acc else acc[key].add(r, fill_value=0)
                if with_stratum:   # also the un-stratified ('all') aggregate
                    g = ["arm"] + keys_extra + ["case_id", "threshold"]
                    r = s.groupby(g, observed=True)[f"dice_v{k}"].agg(["sum", "count"])
                    key = (k, tuple(g))
                    acc[key] = r if key not in acc else acc[key].add(r, fill_value=0)
    frames = []
    for (k, g), r in acc.items():
        r = r.reset_index()
        r["dice"] = r["sum"] / r["count"]
        r["n"] = r["count"].astype(int)
        r["k"] = k
        if "stratum" not in r:
            r["stratum"] = "all"
        frames.append(r.drop(columns=["sum", "count"]))
    return pd.concat(frames, ignore_index=True)


def main():
    # ---- instances (cohort figure) ------------------------------------------------
    fr = []
    for p in sorted(glob.glob(os.path.join(ROOT, "outputs", "analysis_full", "per_sample_free_val_fold*.csv"))):
        f = pd.read_csv(p)
        f = f[f.run.str.startswith("tgtA_fold")]
        fr.append(f[["fold", "case_id", "diameter_mm"] + [f"gt_area_v{k}" for k in K]])
    inst_cv = pd.concat(fr, ignore_index=True)
    inst_cv.to_csv(os.path.join(OUT, "instances_cv.csv"), index=False)
    f = pd.read_csv(os.path.join(ROOT, "outputs", "analysis_test", "per_sample_free_test.csv"))
    inst_te = f[f.run == "tgtA_fold0"][["case_id", "diameter_mm"] + [f"gt_area_v{k}" for k in K]]
    inst_te.to_csv(os.path.join(OUT, "instances_test.csv"), index=False)
    print("instances:", len(inst_cv), len(inst_te), flush=True)

    # ---- cross-validation, case level -------------------------------------------
    cv_paths = sorted(glob.glob(os.path.join(ROOT, "outputs", "analysis_full", "per_sample_val_fold*.csv")))
    cv = agg_chunks(cv_paths, ["fold"], with_stratum=False)
    cv = cv[["arm", "fold", "case_id", "threshold", "k", "dice", "n"]]
    cv.to_csv(os.path.join(OUT, "case_cv.csv.gz"), index=False, float_format="%.6f")
    print("case_cv rows:", len(cv), "arms:", sorted(cv.arm.unique()), flush=True)

    # ---- lesion-unit threshold effect (arm A, V>=4), CV pooled as in case_lesion_sensitivity.py
    les = {}
    tot = {}
    for p in cv_paths:
        for ch in pd.read_csv(p, usecols=USE, chunksize=600_000):
            ch = ch[ch.run.str.startswith("tgtA_fold") & (ch.gt_area_v4 > 0)]
            for t in (0.5, 0.999):
                s = ch[np.isclose(ch.threshold, t)]
                tot.setdefault(t, [0.0, 0])
                tot[t][0] += float(s.dice_v4.sum()); tot[t][1] += int(len(s))
    les["cv_tgtA_v4_lesion_unit"] = {str(t): v[0] / v[1] for t, v in tot.items()}
    les["cv_tgtA_v4_lesion_unit"]["threshold_effect_pts"] = 100 * (tot[0.999][0] / tot[0.999][1] - tot[0.5][0] / tot[0.5][1])
    tot = {}
    for ch in pd.read_csv(os.path.join(ROOT, "outputs", "analysis_test", "per_sample_test.csv"),
                          usecols=USE, chunksize=600_000):
        ch = ch[ch.run.str.startswith("tgtA_fold") & (ch.gt_area_v4 > 0)]
        for t in (0.5, 0.999):
            s = ch[np.isclose(ch.threshold, t)]
            tot.setdefault(t, [0.0, 0])
            tot[t][0] += float(s.dice_v4.sum()); tot[t][1] += int(len(s))
    les["test_tgtA_v4_lesion_unit"] = {str(t): v[0] / v[1] for t, v in tot.items()}
    les["test_tgtA_v4_lesion_unit"]["threshold_effect_pts"] = 100 * (tot[0.999][0] / tot[0.999][1] - tot[0.5][0] / tot[0.5][1])
    json.dump(les, open(os.path.join(OUT, "lesion_unit.json"), "w"), indent=1)
    print("lesion unit:", les, flush=True)

    # ---- held-out test, case level, with size strata --------------------------------
    te = agg_chunks([os.path.join(ROOT, "outputs", "analysis_test", "per_sample_test.csv")], [],
                    with_stratum=True)
    te = te[["arm", "case_id", "stratum", "threshold", "k", "dice", "n"]]
    te.to_csv(os.path.join(OUT, "case_test.csv.gz"), index=False, float_format="%.6f")
    print("case_test rows:", len(te), flush=True)

    # ---- field-of-view ladder, per case (level 2) ------------------------------------
    sc = []
    for ch in pd.read_csv(os.path.join(ROOT, "outputs", "analysis_scope", "scope_test_merged.csv"),
                          usecols=["run", "case_id", "scope", "threshold", "level", "dice"], chunksize=800_000):
        ch = ch[ch.level == 2]
        ch["arm"] = ch.run.str.replace(r"_fold\d+", "", regex=True)
        sc.append(ch.groupby(["arm", "case_id", "scope", "threshold"])["dice"].agg(["sum", "count"]))
    s = sc[0]
    for x in sc[1:]:
        s = s.add(x, fill_value=0)
    s = s.reset_index()
    s["dice"] = s["sum"] / s["count"]
    s[["arm", "case_id", "scope", "threshold", "dice"]].to_csv(os.path.join(OUT, "scope_case.csv.gz"),
                                                              index=False, float_format="%.6f")
    print("scope rows:", len(s), flush=True)


def noise_seeds(cv):
    rows = []
    for arm in ("tgtA", "tgtB"):
        for fold in ("val_fold0", "val_fold1"):
            for k in K:
                vals, vopt = [], []
                for s in (1, 2, 3):
                    d = cv[(cv.arm == f"seed_{arm}_s{s}") & (cv.fold == fold) & (cv.k == k)]
                    g = d.groupby("threshold").dice.mean()
                    vals.append(100 * float(g.loc[0.5])); vopt.append(100 * float(g.max()))
                rows.append(dict(arm=arm, fold=fold, k=k, v=vals, vopt=vopt,
                                 var=float(np.var(vals, ddof=1)), varopt=float(np.var(vopt, ddof=1))))
    json.dump(rows, open(os.path.join(OUT, "noise_seeds.json"), "w"))
    s = np.sqrt(np.mean([r["var"] for r in rows]))
    print(f"noise floor (2 sigma, fixed 0.5): {2 * s:.3f}", flush=True)


QUBIQ = {"QUBIQ-brain-growth": ("qubiq_brain", 7), "QUBIQ-kidney": ("qubiq_kidney", 3),
         "QUBIQ-brain-tumor-task1": ("qubiq_brain_tumor_task01", 3), "QUBIQ-brain-tumor-task2": ("qubiq_brain_tumor_task02", 3),
         "QUBIQ-brain-tumor-task3": ("qubiq_brain_tumor_task03", 3), "QUBIQ-prostate-task1": ("qubiq_prostate_task01", 6),
         "QUBIQ-prostate-task2": ("qubiq_prostate_task02", 6)}


def qubiq_curves():
    """Same aggregation as scripts/analyze_qubiq_all.py (case mean within run, then mean over runs)."""
    out = {}
    for name, (d, n) in QUBIQ.items():
        df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(ROOT, "outputs", "analysis", d, "per_sample_val_fold*.csv")))])
        cols = [f"dice_v{k}" for k in range(1, n + 1)]
        agg = df.groupby(["run", "threshold", "case_id"])[cols].mean().groupby(["run", "threshold"]).mean().reset_index()
        out[name] = {}
        for mode in ("fixed0.5", "opt_thr"):
            sums = {}
            for arm in "ABC":
                sub = agg[agg.run.str.startswith(f"{d}_{arm}_")]
                sums[arm] = [round(100 * (sub.loc[sub.groupby("run")[c].idxmax(), c].mean() if mode == "opt_thr"
                                          else sub.loc[sub.threshold == 0.5, c].mean()), 2) for c in cols]
            out[name][mode] = sums
    json.dump(out, open(os.path.join(OUT, "qubiq_curves.json"), "w"), indent=1)


def gallery_and_example():
    ic = pd.read_csv(os.path.join(OUT, "instances_cv.csv")); ic["set"] = "cv"
    it = pd.read_csv(os.path.join(OUT, "instances_test.csv")); it["set"] = "test"
    inst = pd.concat([ic, it], ignore_index=True)
    inst["idx"] = inst.groupby("case_id").cumcount()            # patch order == instance order (verified)
    inst["ratio"] = inst.gt_area_v4 / inst.gt_area_v1
    inst["stratum"] = np.where(inst.diameter_mm < 5, "3-5", np.where(inst.diameter_mm < 10, "5-10", "10-30"))
    rng = np.random.default_rng(7)
    imgs, votes, meta = [], [], []
    for st in ["3-5", "5-10", "10-30"]:
        for lo, hi in [(0, 0.0001), (0.15, 0.35), (0.35, 0.6), (0.6, 1.01)]:
            s = inst[(inst.stratum == st) & (inst.ratio >= lo) & (inst.ratio < hi)]
            s = s.sample(min(6, len(s)), random_state=int(rng.integers(1e6)))
            for _, r in s.iterrows():
                z = np.load(os.path.join(ROOT, "data", "processed", "patches", f"{r.case_id}.npz"))
                im, v = z["images"][r.idx], z["votes"][r.idx]
                ys, xs = np.nonzero(v >= 2)
                cy, cx = int(round(ys.mean())), int(round(xs.mean()))
                h = 32
                cy = min(max(cy, h), 128 - h); cx = min(max(cx, h), 128 - h)
                imgs.append(im[cy - h:cy + h, cx - h:cx + h]); votes.append(v[cy - h:cy + h, cx - h:cx + h])
                meta.append(dict(case=r.case_id, idx=int(r.idx), set=r.set, stratum=st, bin=lo,
                                 diam=round(float(r.diameter_mm), 2),
                                 a=[int(r.gt_area_v1), int(r.gt_area_v2), int(r.gt_area_v3), int(r.gt_area_v4)],
                                 in_crop=float((v[cy - h:cy + h, cx - h:cx + h] >= 1).sum() / max(1, (v >= 1).sum()))))
    np.savez_compressed(os.path.join(OUT, "gallery_candidates.npz"), imgs=np.array(imgs), votes=np.array(votes))
    json.dump(meta, open(os.path.join(OUT, "gallery_candidates.json"), "w"))
    cid, sl = "LIDC-IDRI-0056", 82                                  # held-out test nodule, 7.2 mm, offset-eligible
    js = json.load(open(os.path.join(ROOT, "data", "processed", "npz", f"{cid}.json")))
    nod = [n for n in js["nodules"] if n["slice"] == sl][0]
    z = np.load(os.path.join(ROOT, "data", "processed", "npz", f"{cid}.npz"))
    np.savez_compressed(os.path.join(OUT, "example_slice.npz"), img=z["slices"][sl], vote=z["consensus"][sl],
                        cy=nod["cy"], cx=nod["cx"], diam=nod["diameter_mm"])


if __name__ == "__main__":
    main()
    noise_seeds(pd.read_csv(os.path.join(OUT, "case_cv.csv.gz")))
    qubiq_curves()
    gallery_and_example()
