# -*- coding: utf-8 -*-
"""make_figure_source_data.py —— 把 Fig. 2 / Fig. 6 需要的曲线从大 CSV 聚合成小表（2026-10-02）。

论文插图脚本 make_figures_v22.py 只读小文件；需要逐样本大 CSV 的两张图（阈值曲线与
预测面积曲线）先由本脚本聚合，结果同时作为"图源数据"随 release 发布。

口径（与 scripts/analyze_symmetric.py 完全一致）：
  - Dice 曲线：GT 非空的样本 -> 病例内对实例取均值 -> 每折对病例取均值 -> 跨折取均值
    （= analyze_symmetric.curve，其 argmax 即 Table 3 的校准阈值）。
  - 预测面积曲线：病例内对实例取均值 -> 对全部病例取均值（= analyze_symmetric 表 5 面积匹配）。
  - GT 参考面积：病例级均值（表 5 的 gt2 = 163.4 px 即此口径）。
  - Brier / 概率质量：threshold-free CSV 逐实例均值（= symmetric_analysis.json threshold_free）。
输入：outputs/analysis_full/per_sample_val_fold{0..4}.csv 及 per_sample_free_val_fold{0..4}.csv
输出：outputs/figures/v22/source_data/cv_threshold_curves.csv
      outputs/figures/v22/source_data/cv_reference_areas.json
"""
import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "outputs", "analysis_full")
OUT = os.path.join(ROOT, "outputs", "figures", "v22", "source_data")
ARMS = ["tgtA", "tgtB", "tgtC", "tgtD"]
LEVELS = (1, 2, 3, 4)


def main():
    os.makedirs(OUT, exist_ok=True)
    cols = ["run", "fold", "case_id", "threshold", "area_pred"] + \
        [f"dice_v{k}" for k in LEVELS] + [f"gt_area_v{k}" for k in LEVELS]
    dice_fold, area_case, gt_case = [], [], []
    files = [f for f in sorted(glob.glob(os.path.join(SRC, "per_sample_val_fold*.csv")))
             if "free" not in os.path.basename(f)]
    for f in files:
        d = pd.read_csv(f, usecols=cols)
        d["arm"] = d.run.str.split("_fold").str[0]
        d = d[d.arm.isin(ARMS) & d.run.str.match(r"^tgt[ABCD]_fold\d$")]
        for k in LEVELS:
            s = d[d[f"gt_area_v{k}"] > 0]
            pc = s.groupby(["arm", "fold", "threshold", "case_id"])[f"dice_v{k}"].mean()
            pf = pc.groupby(["arm", "fold", "threshold"]).mean().rename(f"dice_v{k}")
            dice_fold.append(pf.reset_index().assign(level=k))
        area_case.append(d.groupby(["arm", "threshold", "case_id"]).area_pred.mean().reset_index())
        r0 = sorted(d.run.unique())[0]
        t0 = sorted(d.threshold.unique())[0]
        one = d[(d.run == r0) & np.isclose(d.threshold, t0)]
        gt_case.append(one.groupby("case_id")[[f"gt_area_v{k}" for k in LEVELS]].mean())
        print(f"  {os.path.basename(f)}: {len(d)} rows, cases {d.case_id.nunique()}", flush=True)

    df = pd.concat(dice_fold, ignore_index=True)
    curves = None
    for k in LEVELS:
        c = (df[df.level == k].groupby(["arm", "fold", "threshold"])[f"dice_v{k}"].first()
             .groupby(["arm", "threshold"]).mean().rename(f"dice_v{k}"))
        curves = c.to_frame() if curves is None else curves.join(c, how="outer")
    area = (pd.concat(area_case).groupby(["arm", "threshold", "case_id"]).area_pred.mean()
            .groupby(["arm", "threshold"]).mean().rename("area_case_mean"))
    curves = curves.join(area, how="outer").reset_index()
    curves.to_csv(os.path.join(OUT, "cv_threshold_curves.csv"), index=False, float_format="%.6f")

    gt = pd.concat(gt_case)
    free = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(SRC, "per_sample_free_val_fold*.csv")))],
                     ignore_index=True)
    free["arm"] = free.run.str.split("_fold").str[0]
    free = free[free.arm.isin(ARMS) & free.run.str.match(r"^tgt[ABCD]_fold\d$")]
    ref = {
        "n_cases": int(len(gt)),
        "gt_area_case_mean_px": {f"G{k}": float(gt[f"gt_area_v{k}"].mean()) for k in LEVELS},
        "brier_instance_mean": {a: float(free[free.arm == a].brier.mean()) for a in ARMS},
        "prob_mass_instance_mean": {a: float(free[free.arm == a].prob_mass.mean()) for a in ARMS},
        "n_instances": {a: int((free.arm == a).sum()) for a in ARMS},
    }
    with open(os.path.join(OUT, "cv_reference_areas.json"), "w", encoding="utf-8") as fh:
        json.dump(ref, fh, ensure_ascii=False, indent=2)
    print(json.dumps(ref, indent=1))
    for a in ARMS:
        c = curves[curves.arm == a].set_index("threshold")
        print(a, " ".join(f"opt_v{k}={c[f'dice_v{k}'].idxmax():g}:{c[f'dice_v{k}'].max():.4f}" for k in LEVELS))


if __name__ == "__main__":
    main()
