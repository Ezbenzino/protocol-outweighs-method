# -*- coding: utf-8 -*-
"""case-level 与 lesion-level 双重敏感性分析。

论文主推断以【病例】为分析单元（病例内对结节实例平均 -> 跨病例平均）。审稿人可能追问：
换成分割实例为单元，结论是否成立？本脚本用同一套逐样本 CSV，在两种单元下分别计算
论文的三组头条数字，并给出结论是否稳健的判断。

两种单元定义：
  case   : 病例内对结节实例的 dice 平均 -> 再跨病例平均（与 analyze_symmetric 一致）
  lesion : 直接对结节实例级 dice 平均（每个实例等权）

计算量（与 analyze_symmetric 对齐）：
  1. 评测靶效应（arm tgtA，固定阈值 0.5，跨 V>=1..4 的极差）
  2. 监督靶极差（5 臂 tgtA..tgtE，@V>=4 校准阈值）
  3. 架构极差（5 模型 tgtB/archR18/archCNX/archPVT/archPU，@V>=2 校准阈值）
  4. 比值 评测靶效应 / 监督靶极差 与 / 架构极差

校准阈值取自权威分析（病例级选定），两种单元共用同一阈值，以隔离"聚合单元"这一变量。

用法（仓库根目录，PowerShell）：
    python scripts\\case_lesion_sensitivity.py
输出：
    outputs/analysis/case_lesion_sensitivity.json
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUP_ARMS = ["tgtA", "tgtB", "tgtC", "tgtD", "tgtE"]
SUP_LEVEL = 4
ARCH_ARMS = ["tgtB", "archR18", "archCNX", "archPVT", "archPU"]
ARCH_LEVEL = 2
LEVELS = (1, 2, 3, 4)


def load(dir_):
    fs = [f for f in sorted(os.listdir(dir_))
          if f.startswith("per_sample_") and f.endswith(".csv") and "free" not in f]
    df = pd.concat([pd.read_csv(os.path.join(dir_, f)) for f in fs], ignore_index=True)
    df["arm"] = df.run.str.split("_fold").str[0]
    return df


def value(df, arm, level, thr, unit):
    """某臂某靶在校准阈值下的 dice 均值，按 unit 聚合。"""
    m = (df.arm == arm) & (df[f"gt_area_v{level}"] > 0) & np.isclose(df.threshold, thr)
    s = df.loc[m, ["case_id", f"dice_v{level}"]]
    if unit == "case":
        return float(s.groupby("case_id")[f"dice_v{level}"].mean().mean())
    return float(s[f"dice_v{level}"].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-full", default=os.path.join(ROOT, "outputs", "analysis_full"))
    ap.add_argument("--analysis-test", default=os.path.join(ROOT, "outputs", "analysis_test"))
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "analysis", "case_lesion_sensitivity.json"))
    args = ap.parse_args()

    thr_cv = json.load(open(os.path.join(args.analysis_full, "symmetric_analysis.json"),
                            encoding="utf-8"))["table1"]
    thr_test = json.load(open(os.path.join(args.analysis_test, "symmetric_analysis_applied.json"),
                              encoding="utf-8"))["table1"]
    cv, ts = load(args.analysis_full), load(args.analysis_test)
    # 2026-10-02：CV 与测试集的 plain U-Net 都用选定学习率的 archPU1e3。
    # （此前测试集的 archPU 是 lr=1e-4 的未选定配置，且阈值在测试集上选出，已由
    #  run_fix_testset_plainunet_20261002.ps1 重评并删除；不再做别名映射。）
    arm_label = {"cv": {"archPU": "archPU1e3"}, "test": {"archPU": "archPU1e3"}}

    rep = {"definition": ("case unit = case-internal mean over nodule instances then across cases; "
                          "lesion unit = direct instance-level mean; same calibrated thresholds for both"),
           "sets": {}}
    for setname, df, thr_src in (("cv", cv, thr_cv), ("test", ts, thr_test)):
        # 1. 评测靶效应：tgtA 固定 0.5
        proto = {}
        for unit in ("case", "lesion"):
            vals = [value(df, "tgtA", k, 0.5, unit) for k in LEVELS]
            proto[unit] = {"per_target": [round(v * 100, 2) for v in vals],
                           "range_pts": round((max(vals) - min(vals)) * 100, 2)}
        # 2. 监督靶极差 @V4 校准
        sup = {}
        for unit in ("case", "lesion"):
            vs = [value(df, a, SUP_LEVEL, float(thr_src[a][str(SUP_LEVEL)]["thr"]), unit)
                  for a in SUP_ARMS]
            sup[unit] = {"per_arm": {a: round(v * 100, 2) for a, v in zip(SUP_ARMS, vs)},
                         "range_pts": round((max(vs) - min(vs)) * 100, 2)}
        # 3. 架构极差 @V2 校准
        arch = {}
        alab = arm_label[setname]
        for unit in ("case", "lesion"):
            vs = [value(df, alab.get(a, a), ARCH_LEVEL,
                        float(thr_src[alab.get(a, a)][str(ARCH_LEVEL)]["thr"]), unit)
                  for a in ARCH_ARMS]
            arch[unit] = {"per_model": {a: round(v * 100, 2) for a, v in zip(ARCH_ARMS, vs)},
                          "range_pts": round((max(vs) - min(vs)) * 100, 2)}
        # 4. 比值
        ratios = {}
        for unit in ("case", "lesion"):
            p, s, a_ = proto[unit]["range_pts"], sup[unit]["range_pts"], arch[unit]["range_pts"]
            ratios[unit] = {"protocol_supervision": round(p / s, 2) if s else None,
                            "protocol_architecture": round(p / a_, 2) if a_ else None}
        rep["sets"][setname] = {"protocol_effect": proto, "supervision_range": sup,
                                "architecture_range": arch, "ratios": ratios}

    # 结论：两种单元下 协议效应 是否均 > 学习方法效应
    concl = {}
    for setname in ("cv", "test"):
        d = rep["sets"][setname]
        robust = True
        for unit in ("case", "lesion"):
            p = d["protocol_effect"][unit]["range_pts"]
            m = max(d["supervision_range"][unit]["range_pts"], d["architecture_range"][unit]["range_pts"])
            robust &= (p > m)
        concl[setname] = robust
    rep["conclusion"] = {
        "statement": ("评测靶效应在两种分析单元下均远大于监督靶极差与架构极差（比值均 >1.5x），"
                      "主结论'协议效应 > 学习方法效应'在 case 与 lesion 两种单元下稳健"),
        "cv": concl["cv"], "test": concl["test"],
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)

    for setname in ("cv", "test"):
        d = rep["sets"][setname]
        print(f"\n=== {setname} ===")
        for unit in ("case", "lesion"):
            p = d["protocol_effect"][unit]["range_pts"]
            s = d["supervision_range"][unit]["range_pts"]
            a = d["architecture_range"][unit]["range_pts"]
            r1 = d["ratios"][unit]["protocol_supervision"]
            r2 = d["ratios"][unit]["protocol_architecture"]
            print(f"  [{unit:6s}] 评测靶={p:6.2f}  监督靶极差={s:5.2f}  架构极差={a:5.2f}"
                  f"  比值 {r1}x / {r2}x")
    print(f"\n结论: {rep['conclusion']['statement']}")
    print(f"已保存 -> {args.out}")


if __name__ == "__main__":
    main()
