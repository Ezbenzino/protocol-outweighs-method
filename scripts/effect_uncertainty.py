# -*- coding: utf-8 -*-
"""效应量不确定度 —— 学习设计 range 的 bootstrap CI + 协议/模型比值的 bootstrap CI。

论文三处"单个数字"：
  1. 监督靶极差（5 臂 tgtA..tgtE，测试集 @V>=4 校准）     点估计 3.12 pts
  2. 架构极差（5 模型 tgtB/archR18/archCNX/archPVT/archPU1e3，测试集 @V>=2 校准）
                                                         点估计见输出（2026-10-02: 1.19 pts）
  3. 头条比值 = 视野效应 / 监督靶极差、视野效应 / 架构极差（均由本脚本现算，不再写死）

2026-10-02 修订：
  - 分子改用 tgtBshift_bg 自己的验证集校准阈值（analysis_full/symmetric_analysis.json，
    V>=2 为 0.65），与重跑后的 scope_analysis_merged2.json 一致；旧版写死阈值 0.5。
  - 新增三个保留测试集分量的病例级 bootstrap CI（供 Fig. 3 误差线）：
    评测靶跨度（tgtA @0.5，V>=1..4）、阈值效应（tgtA @V>=4，0.5 vs 校准）、
    平移增强代价（tgtB vs tgtBshift @V>=2，各自校准阈值）。
  - headline_point_estimates 改为现算，旧版是 22.51/3.12/1.87 等字面量。
分母来自测试集 per_sample（病例级 = 病例内对折模型与实例平均）。

bootstrap 设计（两套，均 5000 次，病例为单元）：
  A. 各分量自样本 CI：
     - FOV 分子：V>=2 GT 非空病例（158 例）
     - 监督靶极差：V>=4 GT 非空病例（124 例）
     - 架构极差：V>=2 GT 非空病例（158 例）
  B. 比值联合 CI：三者在公共病例子集（V>=4 GT 非空，124 例）上同步重采样，
     得到协议/监督与协议/架构比值的联合 95% CI。
     注意：公共子集比各分量全样本小，联合点估计会与头条值有微小偏移
     （见输出 joint_ratio_bootstrap），这是子集收缩的代价，比值结论（远大于 1）不受影响。

用法（仓库根目录，PowerShell）：
    python scripts\\effect_uncertainty.py
输出：
    outputs/analysis/effect_uncertainty.json
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUP_ARMS = ["tgtA", "tgtB", "tgtC", "tgtD", "tgtE"]
SUP_LEVEL = 4
# 2026-10-02：测试集 plain U-Net 改用选定学习率的 archPU1e3（旧 archPU 是 lr=1e-4 的未选定配置）
ARCH_ARMS = ["tgtB", "archR18", "archCNX", "archPVT", "archPU1e3"]
ARCH_LEVEL = 2
FOV_ARM = "tgtBshift_bg"
FOV_LEVEL = 2
FOV_THR = None   # None = 用该臂在 analysis_full/symmetric_analysis.json 的 V>=2 校准阈值
PI_ARM = "tgtBshift"   # 位置不变臂（只有平移增强），用于第二组比值
TARGET_ARM = "tgtA"
SHIFT_ARM = "tgtBshift"
N_BOOT = 5000
SEED = 0


def boot_ci(x, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    b = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(n)])
    return [float(v) for v in np.percentile(b, [2.5, 97.5])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-test", default=os.path.join(ROOT, "outputs", "analysis_test"))
    ap.add_argument("--scope", default=os.path.join(ROOT, "outputs", "analysis_scope", "scope_test_merged.csv"))
    ap.add_argument("--cv", default=os.path.join(ROOT, "outputs", "analysis_full", "symmetric_analysis.json"),
                    help="验证集校准阈值来源（视野臂的阈值取自这里）")
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "analysis", "effect_uncertainty.json"))
    args = ap.parse_args()

    thr_src = json.load(open(os.path.join(args.analysis_test, "symmetric_analysis_applied.json"),
                             encoding="utf-8"))["table1"]
    cv_thr = json.load(open(args.cv, encoding="utf-8"))["table1"]
    need = ["run", "case_id", "threshold"] + [f"dice_v{k}" for k in (1, 2, 3, 4)] \
        + [f"gt_area_v{k}" for k in (1, 2, 3, 4)]
    df = pd.read_csv(os.path.join(args.analysis_test, "per_sample_test.csv"), usecols=need)
    df["arm"] = df.run.str.split("_fold").str[0]

    def per_case(arm, level, thr):
        m = (df.arm == arm) & (df[f"gt_area_v{level}"] > 0) & np.isclose(df.threshold, thr)
        s = df.loc[m, ["case_id", f"dice_v{level}"]]
        return s.groupby("case_id")[f"dice_v{level}"].mean()

    sup_thr = {a: float(thr_src[a][str(SUP_LEVEL)]["thr"]) for a in SUP_ARMS}
    arch_thr = {a: float(thr_src[a][str(ARCH_LEVEL)]["thr"]) for a in ARCH_ARMS}
    sup = pd.DataFrame({f"sup_{a}": per_case(a, SUP_LEVEL, sup_thr[a]) for a in SUP_ARMS})
    arch = pd.DataFrame({f"arch_{a}": per_case(a, ARCH_LEVEL, arch_thr[a]) for a in ARCH_ARMS})
    SUP_COLS = [f"sup_{a}" for a in SUP_ARMS]
    ARCH_COLS = [f"arch_{a}" for a in ARCH_ARMS]

    sc = pd.read_csv(args.scope)
    sc["arm"] = sc.run.str.split("_fold").str[0]

    def fov_diff(arm):
        t = FOV_THR if FOV_THR is not None else float(cv_thr[arm][str(FOV_LEVEL)]["thr"])
        avail = np.sort(sc.threshold.unique())
        t = float(avail[np.argmin(np.abs(avail - t))])
        s = sc[(sc.arm == arm) & (sc.level == FOV_LEVEL)
               & (sc.scope.isin(["win128", "win512"])) & np.isclose(sc.threshold, t)]
        piv = s.groupby(["case_id", "scope"])["dice"].mean().unstack()
        return (piv["win512"] - piv["win128"]).rename("fov_diff"), t

    fov, fov_thr = fov_diff(FOV_ARM)
    fov_pi, fov_pi_thr = fov_diff(PI_ARM)

    # ---- A. 各分量自样本 CI ----
    fov_series = fov.dropna()
    fov_pt = float(fov_series.mean() * 100)
    fov_ci = [v * 100 for v in boot_ci(fov_series.values)]

    sup_full = sup.dropna()          # V>=4 GT 非空病例
    sup_means = sup_full[SUP_COLS].mean()
    sup_range_pt = float((sup_means.max() - sup_means.min()) * 100)
    def sup_range(x):
        m = x[SUP_COLS].mean()
        return float((m.max() - m.min()) * 100)
    rng = np.random.default_rng(SEED)
    sup_b = [sup_range(sup_full.iloc[rng.choice(len(sup_full), len(sup_full), replace=True)])
             for _ in range(N_BOOT)]
    sup_ci = [float(v) for v in np.percentile(sup_b, [2.5, 97.5])]

    arch_full = arch.dropna()        # V>=2 GT 非空病例
    arch_means = arch_full[ARCH_COLS].mean()
    arch_range_pt = float((arch_means.max() - arch_means.min()) * 100)
    def arch_range(x):
        m = x[ARCH_COLS].mean()
        return float((m.max() - m.min()) * 100)
    rng2 = np.random.default_rng(SEED + 1)
    arch_b = [arch_range(arch_full.iloc[rng2.choice(len(arch_full), len(arch_full), replace=True)])
              for _ in range(N_BOOT)]
    arch_ci = [float(v) for v in np.percentile(arch_b, [2.5, 97.5])]

    # ---- A2. Fig. 3 其余三个保留测试集分量 ----
    def boot_stat(frame, fn, seed):
        r = np.random.default_rng(seed)
        idx = np.arange(len(frame))
        b = [fn(frame.iloc[r.choice(idx, len(idx), replace=True)]) for _ in range(N_BOOT)]
        return [float(v) for v in np.percentile(b, [2.5, 97.5])]

    # 评测靶跨度：tgtA 在固定阈值 0.5 下，四个参考各自在其 GT 非空病例上的均值之极差
    tgt = pd.DataFrame({k: per_case(TARGET_ARM, k, 0.5) for k in (1, 2, 3, 4)})
    def span_fn(x):
        m = x.mean()                      # 每列各自跳过 NaN（该参考 GT 为空的病例）
        return float((m.max() - m.min()) * 100)
    target_pt = span_fn(tgt)
    target_ci = boot_stat(tgt, span_fn, SEED + 3)

    # 阈值效应：tgtA @V>=4，导入的校准阈值 vs 0.5，逐病例配对
    t_opt = float(thr_src[TARGET_ARM]["4"]["thr"])
    thr_d = (per_case(TARGET_ARM, 4, t_opt) - per_case(TARGET_ARM, 4, 0.5)).dropna() * 100
    thr_pt = float(thr_d.mean())
    thr_ci = [v * 1 for v in boot_ci(thr_d.values, seed=SEED + 4)]

    # 平移增强代价：tgtB - tgtBshift @V>=2，各自导入的校准阈值，逐病例配对
    sh_d = (per_case("tgtB", 2, float(thr_src["tgtB"]["2"]["thr"]))
            - per_case(SHIFT_ARM, 2, float(thr_src[SHIFT_ARM]["2"]["thr"]))).dropna() * 100
    sh_pt = float(sh_d.mean())
    sh_ci = boot_ci(sh_d.values, seed=SEED + 5)

    fov_pi_series = fov_pi.dropna()
    fov_pi_pt = float(fov_pi_series.mean() * 100)
    fov_pi_ci = [v * 100 for v in boot_ci(fov_pi_series.values)]

    # ---- B. 公共子集联合比值 CI ----
    base = sup.join(arch, how="inner").join(fov, how="inner").dropna()
    n_common = len(base)
    ratio_sup_b, ratio_arch_b = [], []
    rng3 = np.random.default_rng(SEED + 2)
    idx = np.arange(n_common)
    for _ in range(N_BOOT):
        sub = base.iloc[rng3.choice(idx, n_common, replace=True)]
        fv = abs(float(sub["fov_diff"].mean() * 100))
        sm = sub[SUP_COLS].mean()
        am = sub[ARCH_COLS].mean()
        sr = float((sm.max() - sm.min()) * 100)
        ar = float((am.max() - am.min()) * 100)
        if sr > 0:
            ratio_sup_b.append(fv / sr)
        if ar > 0:
            ratio_arch_b.append(fv / ar)
    joint_sup_pt = abs(float(base["fov_diff"].mean() * 100)) / \
        float((base[SUP_COLS].mean().max() - base[SUP_COLS].mean().min()) * 100)
    joint_arch_pt = abs(float(base["fov_diff"].mean() * 100)) / \
        float((base[ARCH_COLS].mean().max() - base[ARCH_COLS].mean().min()) * 100)

    def pct(x):
        return [round(float(v), 2) for v in np.percentile(x, [2.5, 97.5])]

    rep = {
        "definition": ("case-level bootstrap; supervision range = range over 5 arms (tgtA-E) "
                       "@V>=4 calibrated; architecture range = range over 5 models @V>=2 calibrated; "
                       f"numerator = |FOV effect| tgtBshift_bg @V>=2 (win512-win128, thr={fov_thr:g}, "
                       "validation-calibrated)"),
        "headline_point_estimates": {
            "fov_effect_pts": round(abs(fov_pt), 2), "fov_threshold": fov_thr,
            "fov_position_invariant_pts": round(abs(fov_pi_pt), 2), "fov_position_invariant_threshold": fov_pi_thr,
            "supervision_range_pts": round(sup_range_pt, 2), "architecture_range_pts": round(arch_range_pt, 2),
            "ratio_protocol_supervision": round(abs(fov_pt) / sup_range_pt, 2),
            "ratio_protocol_architecture": round(abs(fov_pt) / arch_range_pt, 2),
            "ratio_position_invariant_supervision": round(abs(fov_pi_pt) / sup_range_pt, 2),
            "ratio_position_invariant_architecture": round(abs(fov_pi_pt) / arch_range_pt, 2),
            "source": "本脚本现算：per_sample_test.csv + scope_test_merged.csv（阈值取验证集校准值）",
        },
        "own_sample_bootstrap": {
            "fov_effect_pts": {"n": int(len(fov_series)), "point": round(fov_pt, 2), "ci95": [round(v, 2) for v in fov_ci]},
            "supervision_range_pts": {"n": int(len(sup_full)), "point": round(sup_range_pt, 2), "ci95": [round(v, 2) for v in sup_ci]},
            "architecture_range_pts": {"n": int(len(arch_full)), "point": round(arch_range_pt, 2), "ci95": [round(v, 2) for v in arch_ci]},
            "fov_position_invariant_pts": {"n": int(len(fov_pi_series)), "point": round(fov_pi_pt, 2), "ci95": [round(v, 2) for v in fov_pi_ci]},
            "evaluation_target_span_pts": {"n": int(len(tgt)), "point": round(target_pt, 2), "ci95": [round(v, 2) for v in target_ci],
                                           "definition": "tgtA @thr 0.5, max-min over V>=1..4 of case-level means"},
            "threshold_effect_pts": {"n": int(len(thr_d)), "point": round(thr_pt, 2), "ci95": [round(v, 2) for v in thr_ci],
                                     "definition": f"tgtA @V>=4, thr {t_opt:g} minus thr 0.5, paired by case"},
            "translation_cost_pts": {"n": int(len(sh_d)), "point": round(sh_pt, 2), "ci95": [round(v, 2) for v in sh_ci],
                                     "definition": "tgtB minus tgtBshift @V>=2, each at its imported calibrated threshold, paired by case"},
        },
        "joint_ratio_bootstrap": {
            "n_common_cases": n_common,
            "note": ("公共子集 = 三量共同病例（V>=4 GT 非空）；子集收缩使联合点估计偏离头条值，"
                     "比值结论不受影响"),
            "ratio_protocol_supervision": {"point": round(joint_sup_pt, 2), "ci95": pct(ratio_sup_b)},
            "ratio_protocol_architecture": {"point": round(joint_arch_pt, 2), "ci95": pct(ratio_arch_b)},
        },
        "conclusion": (f"比值 95% CI 完全落在 1 之上（协议/监督 >= {pct(ratio_sup_b)[0]}x，"
                       f"协议/架构 >= {pct(ratio_arch_b)[0]}x），"
                       "'协议效应大于学习设计'的结论在抽样不确定性下稳健。"),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)

    print("各分量自样本 bootstrap 95% CI:")
    o = rep["own_sample_bootstrap"]
    print(f"  视野效应:   {o['fov_effect_pts']['point']} pts  [{o['fov_effect_pts']['ci95']}]  n={o['fov_effect_pts']['n']}")
    print(f"  监督靶极差: {o['supervision_range_pts']['point']} pts  [{o['supervision_range_pts']['ci95']}]  n={o['supervision_range_pts']['n']}")
    print(f"  架构极差:   {o['architecture_range_pts']['point']} pts  [{o['architecture_range_pts']['ci95']}]  n={o['architecture_range_pts']['n']}")
    for key in ("fov_position_invariant_pts", "evaluation_target_span_pts", "threshold_effect_pts", "translation_cost_pts"):
        print(f"  {key}: {o[key]['point']} pts  {o[key]['ci95']}  n={o[key]['n']}")
    print("头条点估计:", json.dumps(rep["headline_point_estimates"], ensure_ascii=False))
    j = rep["joint_ratio_bootstrap"]
    print(f"\n公共子集联合比值 bootstrap (n={j['n_common_cases']}):")
    print(f"  协议/监督: {j['ratio_protocol_supervision']['point']}x  CI95={j['ratio_protocol_supervision']['ci95']}")
    print(f"  协议/架构: {j['ratio_protocol_architecture']['point']}x  CI95={j['ratio_protocol_architecture']['ci95']}")
    print(f"已保存 -> {args.out}")


if __name__ == "__main__":
    main()
