"""对称评测结果分析 v3 —— 纯读 CSV，不碰 GPU，可反复重跑。

v3 相对 v2 的改动：从"两个臂硬编码"改成"任意多个实验臂"，
以支持四个训练靶（tgtA/B/C/D）与后续的架构对照臂。

实验臂由 run 名自动识别：run = "<arm>_fold<k>"，取 "_fold" 之前的部分作为臂名。

输出：
  表 1  实验臂 x 评测靶 的完整交叉矩阵（固定 0.5 与各自校准阈值两套）
  表 2  关键成对比较的病例级配对检验
  表 3  协议效应 vs 模型效应（论文核心表）
  表 4  可报告区间
  表 5  面积匹配
  表 6  跨折稳定性

分析单元是【病例】不是【样本】：同一病例内多个结节高度相关，
按样本做检验会把 n 虚增数倍，p 值假性变小。

用法：
    python scripts\\analyze_symmetric.py
    python scripts\\analyze_symmetric.py --arms tgtA,tgtB,tgtC,tgtD
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

LEVELS = (1, 2, 3, 4)
LVL = {1: "V>=1 union", 2: "V>=2 majority", 3: "V>=3 high", 4: "V>=4 unanim."}
LVL_S = {1: "V>=1", 2: "V>=2", 3: "V>=3", 4: "V>=4"}
W = 96

# 臂名 -> 论文中的标签。未列出的臂会原样使用其 run 前缀。
ARM_LABEL = {
    "tgtA": "A Union (V>=1)",
    "tgtB": "B Majority (V>=2 hard)",
    "tgtC": "C Consensus (V>=2+soft)",
    "tgtD": "D Soft only (p=V/4)",
    "archR18": "ResNet-18 U-Net",
    "archPU": "Plain U-Net (lr 1e-4, not selected)",
    "archPU1e3": "Plain U-Net",
    "main": "[legacy] 6-loss consensus",
    "union": "[legacy] union (CSL leak)",
}
# 需要做配对检验的成对比较：(臂1, 臂2, 这一对回答什么问题)
PAIRS = [
    ("tgtC", "tgtA", "consensus vs union  —— 论文原始对比"),
    ("tgtB", "tgtA", "majority vs union   —— 只变掩膜范围"),
    ("tgtC", "tgtB", "consensus vs majority —— 只变软/硬监督（关键）"),
    ("tgtD", "tgtC", "soft-only vs consensus —— 去掉硬掩膜项"),
    ("tgtD", "tgtB", "soft-only vs majority —— 纯软 vs 纯硬"),
]
# 模型侧效应量参照（来自本项目旧实验，单次运行；架构对照臂跑完后由脚本自动替换）
MODEL_FALLBACK = [
    ("Architecture: plain U-Net vs ResNet-34+ImageNet [legacy]", 0.8789 - 0.8518),
    ("Loss design: 6-term vs 5-term (BRBC removed) [legacy]", 0.8834 - 0.8789),
    ("Loss design: range over 4 weight settings [legacy]", 0.8768 - 0.8751),
    ("Architecture: ResNet-18 vs ResNet-34 [legacy]", 0.0004),
]


def load(out_dir):
    fs = [f for f in sorted(glob.glob(os.path.join(out_dir, "per_sample_*.csv")))
          if "free" not in os.path.basename(f)]
    if not fs:
        raise SystemExit(f"{out_dir} 下没有 per_sample_*.csv，请先跑 eval_symmetric.py")
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    if "fold" not in df.columns:
        df["fold"] = "val_fold0"
    if "gt_area_v2" not in df.columns:
        raise SystemExit("CSV 缺 GT 面积列，请用 eval_symmetric.py v2 重新导出。")
    df["arm"] = df.run.str.split("_fold").str[0]
    ff = sorted(glob.glob(os.path.join(out_dir, "per_sample_free_*.csv")))
    free = pd.concat([pd.read_csv(f) for f in ff], ignore_index=True) if ff else None
    if free is not None:
        free["arm"] = free.run.str.split("_fold").str[0]
    return df, free


def valid(df, arm, k, thr=None):
    """某臂、某评测靶下 GT 非空的样本（GT 为空时 Dice 无定义）。"""
    m = (df.arm == arm) & (df[f"gt_area_v{k}"] > 0)
    if thr is not None:
        m &= np.isclose(df.threshold, thr)
    return df[m]


def curve(df, arm, k):
    """阈值-Dice 曲线。分析单元 = 病例：先在病例内对结节实例平均，
    再按折平均，最后跨折平均。直接对行取均值会让结节多的病例权重高十几倍
    （每例中位 9 个实例、最多 79 个），且与 paired() 的病例级检验口径不一致。"""
    s = valid(df, arm, k)
    per_case = s.groupby(["fold", "threshold", "case_id"])[f"dice_v{k}"].mean()
    return per_case.groupby(["fold", "threshold"]).mean().groupby("threshold").mean()


def paired(df, best, a, b, k):
    """病例级配对检验 + bootstrap CI。两臂各用自己在该评测靶上的校准阈值。"""
    ga = valid(df, a, k, best[(a, k)][0]).groupby(["fold", "case_id"])[f"dice_v{k}"].mean()
    gb = valid(df, b, k, best[(b, k)][0]).groupby(["fold", "case_id"])[f"dice_v{k}"].mean()
    ga, gb = ga.align(gb, join="inner")
    d = (ga - gb).values
    rng = np.random.default_rng(0)
    lo, hi = np.percentile([rng.choice(d, len(d), replace=True).mean() for _ in range(5000)],
                           [2.5, 97.5])
    return dict(n=int(len(d)), diff=float(d.mean()), ci=[float(lo), float(hi)],
                p_t=float(stats.ttest_rel(ga, gb).pvalue),
                p_w=float(stats.wilcoxon(ga, gb).pvalue),
                dz=float(d.mean() / d.std(ddof=1)), win=float((d > 0).mean()))


def fmt_p(p):
    return f"{p:.2e}" if p < 1e-3 else f"{p:.4f}"


def label(a):
    return ARM_LABEL.get(a, a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/analysis")
    ap.add_argument("--arms", default=None, help="逗号分隔，指定顺序；默认用 CSV 中出现的全部臂")
    ap.add_argument("--thr-from", default=None,
                    help="套用另一次分析（通常是验证集）选定的阈值，而不是在本数据上重选。"
                         "指向该次的 symmetric_analysis.json。用于测试集无偏评估。")
    args = ap.parse_args()

    df, free = load(args.out_dir)
    present = list(df.arm.unique())
    arms = ([a.strip() for a in args.arms.split(",") if a.strip() in present] if args.arms
            else [a for a in ARM_LABEL if a in present] + [a for a in present if a not in ARM_LABEL])
    df = df[df.arm.isin(arms)]
    folds = sorted(df.fold.unique())
    n_cases = int(df.groupby("fold").case_id.nunique().sum())
    thr_lo, thr_hi = float(df.threshold.min()), float(df.threshold.max())

    # 阈值来源：默认在本数据上选（验证集用法）；给了 --thr-from 就套用外部阈值。
    # 测试集必须走后者——在测试集上重选阈值就不是无偏估计了。
    imported = None
    if args.thr_from:
        with open(args.thr_from, encoding="utf-8") as fh:
            imported = json.load(fh)["table1"]
        print(f"[阈值来源] 套用 {args.thr_from} 中验证集选定的阈值，本数据上不重选。")

    best = {}
    for a in arms:
        for k in LEVELS:
            c = curve(df, a, k)
            if imported is not None and a in imported and str(k) in imported[a]:
                t = float(imported[a][str(k)]["thr"])
                idx = int(np.argmin(np.abs(np.asarray(c.index, float) - t)))
                best[(a, k)] = (float(c.index[idx]), float(c.iloc[idx]), float(c.loc[0.5]))
            else:
                best[(a, k)] = (float(c.idxmax()), float(c.max()), float(c.loc[0.5]))

    print("=" * W)
    print(f"对称评测分析 v3  |  {len(folds)} 折  {n_cases} 病例  "
          f"{len(valid(df, arms[0], 1, 0.5))} 样本  阈值 {thr_lo}~{thr_hi}")
    print(f"实验臂: {', '.join(label(a) for a in arms)}")
    print("=" * W)
    rep = {"folds": folds, "n_cases": n_cases, "arms": arms}

    # ---------- 表 1 完整交叉矩阵 ----------
    print("\n表 1  实验臂 x 评测靶 交叉矩阵（GT 非空样本；括号内为该臂在该靶上的校准阈值）")
    print("-" * W)
    print(f"{'训练臂':<26}" + "".join(f"{LVL_S[k]:>17}" for k in LEVELS))
    print(f"{'':26}" + "".join(f"{'@0.5 / @opt':>17}" for _ in LEVELS))
    print("-" * W)
    t1 = {}
    for a in arms:
        row_c = f"{label(a):<26}"
        for k in LEVELS:
            t, v, v05 = best[(a, k)]
            row_c += f"{v05:.4f}/{v:.4f}".rjust(17)
        print(row_c)
        print(f"{'':26}" + "".join(f"({best[(a,k)][0]:.2f})".rjust(17) for k in LEVELS))
        t1[a] = {k: dict(thr=best[(a, k)][0], dice_opt=best[(a, k)][1], dice_05=best[(a, k)][2])
                 for k in LEVELS}
    rep["table1"] = t1
    for (a, k), (t, _, _) in (best.items() if imported is None else []):
        if t in (thr_lo, thr_hi):
            print(f"  [警告] {a} @ V>={k} 的最优阈值落在扫描端点 {t}，真峰值可能在区间外。")

    # ---------- 表 2 成对比较 ----------
    print("\n" + "=" * W)
    print("表 2  关键成对比较（病例级配对检验，两臂各用自己的校准阈值）")
    print("=" * W)
    t2 = {}
    for a, b, why in PAIRS:
        if a not in arms or b not in arms:
            continue
        print(f"\n  {label(a)}  minus  {label(b)}     [{why}]")
        print(f"    {'评测靶':<14}{'n':>5}{'差值':>9}{'95%CI':>21}{'p(t)':>11}{'dz':>7}{'占优':>8}")
        t2[f"{a}-{b}"] = {}
        for k in LEVELS:
            r = paired(df, best, a, b, k)
            t2[f"{a}-{b}"][k] = r
            ci = f"[{r['ci'][0]*100:+.2f}, {r['ci'][1]*100:+.2f}]"
            print(f"    {LVL[k]:<14}{r['n']:>5}{r['diff']*100:>+9.2f}{ci:>21}"
                  f"{fmt_p(r['p_t']):>11}{r['dz']:>7.3f}{r['win']:>7.1%}")
    rep["table2"] = t2

    # ---------- 表 3 协议效应 vs 模型效应 ----------
    print("\n" + "=" * W)
    print("表 3  协议效应 vs 模型效应 —— 论文核心表")
    print("=" * W)
    eff = []
    for a in arms:
        if a.startswith("arch"):
            continue
        raw = [best[(a, k)][2] for k in LEVELS]
        cal = [best[(a, k)][1] for k in LEVELS]
        eff.append((f"Evaluation target V>=1..4, {label(a)} (fixed thr 0.5)", "Protocol",
                    max(raw) - min(raw)))
        eff.append((f"Evaluation target V>=1..4, {label(a)} (calibrated)", "Protocol",
                    max(cal) - min(cal)))
        for k in LEVELS:
            t, v, v05 = best[(a, k)]
            if v - v05 > 0.005:
                eff.append((f"Threshold, {label(a)} @ V>={k} (0.5 vs {t:.2f})", "Protocol", v - v05))
    # 训练靶效应 = 同一评测靶下各训练臂之间的极差
    tgts = [a for a in arms if a.startswith("tgt")]
    for k in LEVELS:
        if len(tgts) >= 2:
            vs = [best[(a, k)][1] for a in tgts]
            eff.append((f"Training target (range over {len(tgts)} arms) @ V>={k}, calibrated",
                        "Method", max(vs) - min(vs)))
    # 架构效应：若架构臂已跑完则用实测，否则用旧实验作参照
    archs = [a for a in arms if a.startswith("arch")] + (["tgtB"] if "tgtB" in arms else [])
    if len([a for a in archs if a.startswith("arch")]) >= 1:
        for k in (2,):
            vs = [best[(a, k)][1] for a in archs]
            eff.append((f"Architecture (range over {len(archs)} models) @ V>={k}, calibrated",
                        "Model", max(vs) - min(vs)))
    else:
        eff += [(n, "Model", v) for n, v in MODEL_FALLBACK]
        print("  [注] 架构对照臂尚未跑完，模型侧效应量暂用旧实验（单次运行）作数量级参照。")

    eff.sort(key=lambda r: -r[2])
    print(f"\n{'因素':<62}{'类型':>10}{'效应量':>12}")
    print("-" * W)
    for n, t, v in eff:
        print(f"{n:<62}{t:>10}{v*100:>10.2f} 点" + ("  ***" if t == "Protocol" else ""))
    p_max = max(v for _, t, v in eff if t == "Protocol")
    m_max = max(v for _, t, v in eff if t in ("Model", "Method"))
    print("-" * W)
    print(f"最大协议效应 {p_max*100:.2f} 点  vs  最大模型/方法效应 {m_max*100:.2f} 点"
          f"  ->  {p_max/m_max:.1f}x")
    rep["table3"] = [{"factor": n, "type": t, "effect": v} for n, t, v in eff]
    rep["protocol_vs_model"] = {"protocol_max": p_max, "model_max": m_max, "ratio": p_max / m_max}

    # ---------- 表 4 可报告区间 ----------
    print("\n" + "=" * W)
    print("表 4  可报告区间：同一个模型仅靠选协议能把 Dice 报到多少")
    print("=" * W)
    print(f"{'训练臂':<26}{'固定0.5 跨4靶':>22}{'各靶最优阈值':>22}")
    print(f"{'':26}{'最低~最高 (区间)':>22}{'最低~最高 (区间)':>22}")
    t4 = {}
    for a in arms:
        raw = [best[(a, k)][2] for k in LEVELS]
        cal = [best[(a, k)][1] for k in LEVELS]
        t4[a] = {"fixed05": [min(raw), max(raw), max(raw) - min(raw)],
                 "calibrated": [min(cal), max(cal), max(cal) - min(cal)]}
        print(f"{label(a):<26}"
              f"{f'{min(raw):.4f}~{max(raw):.4f} ({max(raw)-min(raw):.4f})':>22}"
              f"{f'{min(cal):.4f}~{max(cal):.4f} ({max(cal)-min(cal):.4f})':>22}")
    rep["table4"] = t4

    # ---------- 表 5 面积匹配 ----------
    gt2 = float(df.groupby("case_id").gt_area_v2.mean().mean())
    print("\n" + "=" * W)
    print(f"表 5  面积匹配（GT V>=2 平均面积 = {gt2:.1f} px）")
    print("=" * W)
    print(f"{'训练臂':<26}{'阈值':>8}{'预测面积':>12}{'dice_v2':>11}{'dice_v1':>11}")
    t5 = {}
    for a in arms:
        g = (df[df.arm == a].groupby(["threshold", "case_id"]).area_pred.mean()
             .groupby("threshold").mean().to_frame("area"))
        t = float((g.area - gt2).abs().idxmin())
        s = df[(df.arm == a) & np.isclose(df.threshold, t)].groupby("case_id")
        t5[a] = dict(threshold=t, area=float(s.area_pred.mean().mean()),
                     dice_v2=float(s.dice_v2.mean().mean()),
                     dice_v1=float(s.dice_v1.mean().mean()))
        print(f"{label(a):<26}{t:>8.2f}{t5[a]['area']:>12.1f}"
              f"{t5[a]['dice_v2']:>11.4f}{t5[a]['dice_v1']:>11.4f}")
    rep["table5"] = t5

    # ---------- 表 6 跨折稳定性 ----------
    print("\n" + "=" * W)
    print("表 6  跨折稳定性（各臂在 V>=2 上、用自己的校准阈值）")
    print("=" * W)
    t6 = {}
    for a in arms:
        t = best[(a, 2)][0]
        per = (valid(df, a, 2, t).groupby(["fold", "case_id"]).dice_v2.mean()
               .groupby("fold").mean())
        t6[a] = per.to_dict()
        print(f"  {label(a):<26}" + "  ".join(f"{v:.4f}" for v in per.values)
              + f"   -> {per.mean():.4f} +/- {per.std(ddof=1):.4f}")
    rep["table6"] = t6

    # ---------- 与阈值无关的参考指标 ----------
    if free is not None:
        print("\n" + "=" * W)
        print("参考  与阈值无关的指标（靶 p=V/4 是 tgtC/tgtD 的训练目标，")
        print("      因此对它们【不是】中立指标，仅作描述性统计）")
        print("=" * W)
        cols = [c for c in ("soft_dice", "brier", "prob_mass") if c in free.columns]
        f = free[free.arm.isin(arms)].groupby("arm")[cols].mean().reindex(arms)
        f.index = [label(a) for a in arms]
        print(f.to_string())
        print(f"  参考 GT 面积: " + "  ".join(
            f"{LVL_S[k]}={free[f'gt_area_v{k}'].mean():.1f}" for k in LEVELS))
        rep["threshold_free"] = f.to_dict()

    out = os.path.join(args.out_dir,
                       "symmetric_analysis_applied.json" if imported else "symmetric_analysis.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=2, ensure_ascii=False)
    print(f"\n已保存 -> {out}")


if __name__ == "__main__":
    main()
