# -*- coding: utf-8 -*-
"""offset_paired_stats.py —— 位置先验（Section 4.10）的全部正文数字，从 diag_offset*.json 现算。

2026-10-02 新增。此前 §4.10 的配对差 / CI / dz / 聚类 bootstrap 是临时计算后手抄进正文的，
仓库里没有对应脚本；本脚本把它们固化下来，并在阈值改为各臂验证集校准值后重算。

输入（均由 scripts/diag_offset.py 生成，阈值取 outputs/analysis_full/symmetric_analysis.json）：
  outputs/analysis_scope/diag_offset_allarms.json   11 臂 × fold0，76 个合格结节
  outputs/analysis_scope/diag_offset.json           tgtB / tgtBshift × 5 折
输出：
  outputs/analysis_scope/offset_paired_stats.json

口径：
  - 十臂摘要：fold0 模型，d=0 与 d=8 / d=56 的均值差（Dice 点），取中位数与范围。
  - 五折配对：每个结节先对 5 个折模型取均值，再做 tgtBshift - tgtB 的逐结节配对差；
    95% CI 为 5000 次 bootstrap 百分位（numpy default_rng(0)，与原手算一致），dz = 均值/标准差。
  - 76 个结节来自 76 个不同病例（每例一个），因此病例级聚类 bootstrap 与结节级 bootstrap 相同。
用法：
    python scripts/offset_paired_stats.py
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALL = os.path.join(ROOT, "outputs", "analysis_scope", "diag_offset_allarms.json")
FIVE = os.path.join(ROOT, "outputs", "analysis_scope", "diag_offset.json")
OUT = os.path.join(ROOT, "outputs", "analysis_scope", "offset_paired_stats.json")
NOMINAL_MM_PER_PX = 0.6     # 与 preprocess/rebuild_meta.py 的 SPACING_EST 一致
PAIR_D = 40
N_BOOT = 5000


def main():
    a = json.load(open(ALL, encoding="utf-8"))
    f = json.load(open(FIVE, encoding="utf-8"))
    offs = [int(o) for o in a["offsets"]]

    # ---- 十臂（fold0）摘要 ----
    arms = {}
    for run, r in a["runs"].items():
        m = r["mean_by_offset"]
        arms[run] = dict(thr=r["thr"], n=r["n"], d0=m["0"], d8=m["8"], d56=m["56"],
                         drop8_pts=100 * (m["0"] - m["8"]), drop56_pts=100 * (m["0"] - m["56"]))
    noaug = {k: v for k, v in arms.items() if "shift" not in k}
    d8 = np.array([v["drop8_pts"] for v in noaug.values()])
    d56 = np.array([v["drop56_pts"] for v in noaug.values()])
    shift0 = arms.get("tgtBshift_fold0")

    # ---- 五折均值曲线 ----
    def five_mean(arm):
        return {o: float(np.mean([f["runs"][f"{arm}_fold{k}"]["mean_by_offset"][str(o)]
                                  for k in range(5)])) for o in offs}

    curve_b, curve_s = five_mean("tgtB"), five_mean("tgtBshift")

    # ---- d=40 逐结节配对 ----
    def per_nodule(arm, d):
        rows = {}
        for k in range(5):
            for p in f["runs"][f"{arm}_fold{k}"]["per_nodule"]:
                rows.setdefault((p["case_id"], p["slice"]), []).append(p[f"d{d}"])
        return {key: float(np.mean(v)) for key, v in rows.items() if len(v) == 5}

    pb, ps = per_nodule("tgtB", PAIR_D), per_nodule("tgtBshift", PAIR_D)
    keys = sorted(set(pb) & set(ps))
    diff = np.array([ps[k] - pb[k] for k in keys]) * 100
    n = len(diff)
    rng = np.random.default_rng(0)
    boots = [rng.choice(diff, n, replace=True).mean() for _ in range(N_BOOT)]
    ci = [float(v) for v in np.percentile(boots, [2.5, 97.5])]
    cases = sorted({k[0] for k in keys})

    # 合格结节的等效半径（名义 0.6 mm/px）
    dia = {}
    for p in f["runs"]["tgtB_fold0"]["per_nodule"]:
        dia[(p["case_id"], p["slice"])] = p["diameter_mm"]
    radius_px = np.array([dia[k] / NOMINAL_MM_PER_PX / 2 for k in keys])

    rep = {
        "inputs": {"allarms": os.path.relpath(ALL, ROOT), "fivefold": os.path.relpath(FIVE, ROOT),
                   "thr_mode": a.get("thr_mode"), "level": a.get("level")},
        "ten_arms_fold0": {
            "arms": noaug,
            "median_drop8_pts": float(np.median(d8)),
            "range_drop8_pts": [float(d8.min()), float(d8.max())],
            "range_drop56_pts": [float(d56.min()), float(d56.max())],
            "translation_arm_drop8_pts": None if shift0 is None else shift0["drop8_pts"],
        },
        "fivefold_curves": {"tgtB": curve_b, "tgtBshift": curve_s},
        "paired_d40": {
            "contrast": "tgtBshift - tgtB, each nodule averaged over 5 fold models",
            "n_nodules": n, "n_cases": len(cases),
            "mean_pts": float(diff.mean()), "ci95_pts": ci,
            "dz": float(diff.mean() / diff.std(ddof=1)),
            "cluster_bootstrap_note": "one nodule per case -> case-level cluster bootstrap identical to nodule-level",
        },
        "eligible_nodules": {"median_equivalent_radius_px": float(np.median(radius_px)),
                             "nominal_mm_per_px": NOMINAL_MM_PER_PX},
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=2)

    t = rep["ten_arms_fold0"]
    print(f"十臂 d0->d8 中位 {t['median_drop8_pts']:.2f}，范围 {t['range_drop8_pts'][0]:.2f}–{t['range_drop8_pts'][1]:.2f}；"
          f"d0->d56 范围 {t['range_drop56_pts'][0]:.2f}–{t['range_drop56_pts'][1]:.2f}；"
          f"平移臂 d0->d8 {t['translation_arm_drop8_pts']:+.2f}")
    print(f"五折 tgtB: d0 {curve_b[0]:.4f}  d8 {curve_b[8]:.4f}  d56 {curve_b[56]:.4f}")
    p = rep["paired_d40"]
    print(f"d=40 配对差 {p['mean_pts']:+.2f} [{p['ci95_pts'][0]:+.2f}, {p['ci95_pts'][1]:+.2f}]  dz {p['dz']:.2f}  "
          f"n={p['n_nodules']} 结节 / {p['n_cases']} 病例")
    print(f"合格结节等效半径中位数 {rep['eligible_nodules']['median_equivalent_radius_px']:.2f} px")
    print(f"已保存 -> {OUT}")


if __name__ == "__main__":
    main()
