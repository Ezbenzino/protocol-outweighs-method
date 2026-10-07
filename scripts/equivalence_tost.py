# -*- coding: utf-8 -*-
"""TOST 等价性检验 —— 支撑论文 §4.3 的"软监督效应与测量分辨率等价"声明。

论文原文（p130）：
  "Two one-sided tests with an equivalence bound of ±1.19 points — the run-to-run
   noise floor of Section 4.8 — reject non-equivalence for both contrasts, whose
   intervals lie strictly inside the bound."

本脚本正式实现该检验，并对 ±1.0 / ±1.19 / ±1.5 三种界限做敏感性分析，
同时给出固定 0.5 阈值与各臂校准阈值两种口径（噪声底 1.19 本身是 fixed-0.5
pooled 口径，故两种都报告，避免口径混用）。

TOST 定义（配对，two one-sided t-tests，df = n-1，α = 0.05）：
  H0_U: mean >= Δ   vs  H1_U: mean <  Δ     ->  p_U = t_cdf((mean - Δ)/SE)
  H0_L: mean <= -Δ  vs  H1_L: mean >  -Δ    ->  p_L = 1 - t_cdf((mean + Δ)/SE)
  等价成立  <=>  max(p_U, p_L) < α  <=>  (1-2α)CI 完全落在 (-Δ, Δ) 内。

分析单元与 analyze_symmetric.paired() 完全一致：
  病例级 = 病例内对结节实例 dice 平均 -> 跨病例；两臂各用自己的校准阈值。

用法（仓库根目录，PowerShell）：
    python scripts\\equivalence_tost.py
输出：
    outputs/analysis/equivalence_tost.json
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEVELS = (1, 2, 3, 4)
LVL = {1: "V>=1 union", 2: "V>=2 majority", 3: "V>=3 high", 4: "V>=4 unanim."}
# 关键对比：C（软+硬掩膜）minus B（多数硬掩膜）—— 只变软/硬监督
ARM_A, ARM_B = "tgtC", "tgtB"
BOUNDS_POINTS = [1.0, 1.19, 1.5]      # 论文建议的三种界限（Dice points）
ALPHA = 0.05


def load(out_dir):
    fs = [f for f in sorted(glob.glob(os.path.join(out_dir, "per_sample_*.csv")))
          if "free" not in os.path.basename(f)]
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    df["arm"] = df.run.str.split("_fold").str[0]
    return df


def case_dice(df, arm, k, thr):
    """病例级 dice：病例内对结节实例平均；返回 index=case_id 的 Series。"""
    m = (df.arm == arm) & (df[f"gt_area_v{k}"] > 0) & np.isclose(df.threshold, thr)
    s = df[m].groupby(["fold", "case_id"])[f"dice_v{k}"].mean()
    return s


def calibrate(df, arm, k):
    """该臂在该评测靶上的校准阈值（最大化病例级曲线，与 analyze_symmetric 一致）。"""
    m = (df.arm == arm) & (df[f"gt_area_v{k}"] > 0)
    sub = df[m]
    curve = sub.groupby(["fold", "threshold", "case_id"])[f"dice_v{k}"].mean()
    curve = curve.groupby(["fold", "threshold"]).mean().groupby("threshold").mean()
    return float(curve.idxmax())


def tost(d, delta):
    """delta 单位与 d 相同（fraction）。返回 dict。"""
    d = np.asarray(d, float)
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    mean = d.mean()
    df = n - 1
    p_u = float(stats.t.cdf((mean - delta) / se, df)) if se > 0 else float("nan")
    p_l = float(1 - stats.t.cdf((mean + delta) / se, df)) if se > 0 else float("nan")
    p_max = max(p_u, p_l)
    lo, hi = mean - stats.t.ppf(1 - ALPHA, df) * se, mean + stats.t.ppf(1 - ALPHA, df) * se
    return {
        "n": n, "mean_points": mean * 100,
        "se_points": se * 100,
        "ci95_points": [lo * 100, hi * 100],
        "t_upper": float((mean - delta) / se) if se > 0 else None,
        "t_lower": float((mean + delta) / se) if se > 0 else None,
        "p_upper": p_u, "p_lower": p_l,
        "p_max": p_max,
        "equivalent": bool(p_max < ALPHA),
        "ci_within_bound": bool(lo > -delta and hi < delta),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default=os.path.join(ROOT, "outputs", "analysis_full"))
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "analysis", "equivalence_tost.json"))
    args = ap.parse_args()

    df = load(args.analysis)
    # 校准阈值（与 analyze_symmetric 表 1 的括号值应一致）
    best = {(a, k): calibrate(df, a, k) for a in (ARM_A, ARM_B) for k in LEVELS}

    rep = {"definition": "TOST paired two one-sided t-tests; case-level aggregation "
                         "(case-internal mean over instances, then across cases); "
                         "alpha=0.05",
           "contrast": f"{ARM_A} (Consensus, V>=2+soft) minus {ARM_B} (Majority, V>=2 hard)",
           "noise_floor": "1.19 points = 2-sigma pooled, fixed-0.5 (noise_floor.json)",
           "bounds_points": BOUNDS_POINTS,
           "results": {}}

    for k in (1, 2):   # G1 / G2 是等价性声明的两个参考
        # 固定 0.5 阈值口径（与噪声底同一阈值策略）
        gA05 = case_dice(df, ARM_A, k, 0.5)
        gB05 = case_dice(df, ARM_B, k, 0.5)
        gA05, gB05 = gA05.align(gB05, join="inner")
        d05 = (gA05 - gB05).values
        # 各臂校准阈值口径（与论文 p130 的 +0.06/+0.12 数字一致）
        gA = case_dice(df, ARM_A, k, best[(ARM_A, k)])
        gB = case_dice(df, ARM_B, k, best[(ARM_B, k)])
        gA, gB = gA.align(gB, join="inner")
        dcal = (gA - gB).values

        row = {"target": LVL[k], "calibrated_thresholds": {"C": best[(ARM_A, k)], "B": best[(ARM_B, k)]},
               "policies": {}}
        for pol, d in (("fixed0.5", d05), ("calibrated", dcal)):
            pol_row = {"n": len(d), "mean_diff_points": float(d.mean() * 100),
                       "tost_by_bound": {}}
            for b in BOUNDS_POINTS:
                pol_row["tost_by_bound"][f"±{b}"] = tost(d, b / 100.0)
            # 顺带报论文口径的配对 t / CI 复核
            pol_row["paired_ttest_p"] = float(stats.ttest_rel(pd.Series(d), 0).pvalue) if len(d) > 1 else None
            row["policies"][pol] = pol_row
        rep["results"][str(k)] = row

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)

    # 控制台摘要
    print("TOST 等价性检验（C Consensus minus B Majority）")
    for k in (1, 2):
        r = rep["results"][str(k)]
        print(f"\n{r['target']}  (校准阈值 C={r['calibrated_thresholds']['C']}, B={r['calibrated_thresholds']['B']})")
        for pol in ("fixed0.5", "calibrated"):
            pr = r["policies"][pol]
            print(f"  [{pol}] mean={pr['mean_diff_points']:+.2f} pts  n={pr['n']}")
            for b in BOUNDS_POINTS:
                t = pr["tost_by_bound"][f"±{b}"]
                ci = t["ci95_points"]
                print(f"    ±{b}: p_max={t['p_max']:.4f}  CI95=[{ci[0]:+.2f},{ci[1]:+.2f}]  "
                      f"{'等价' if t['equivalent'] else '不等价'}")
    print(f"\n已保存 -> {args.out}")


if __name__ == "__main__":
    main()
