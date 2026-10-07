# -*- coding: utf-8 -*-
"""评测空间（F3）的结果分析：从 patch 到整体积的完整阶梯。

阶梯的每一级只动一个变量，因此差值可以被单独归因：

    patch128 -> win128   只改"模型看到多少上下文"（输入 128 裁剪 vs 完整切片）
    win128   -> win512   只改"评测窗口多大"，模型输入完全不变  <== 纯视野效应
    win512   -> slice    只改"是否只在结节邻域内评测"（512 窗口已接近整片，差值应很小）
    slice    -> volume   只改"是否纳入不含结节的切片"          <== 与训练分布混杂

分析单元是【病例】：patch/win 级在脚本内已按病例对该病例的全部结节求过均值，
slice 级已按病例对含结节切片求过均值，volume 级本来就是每病例一个数。
表 0 的自检额外给出按结节数加权的均值，因为已有的 patch 级结果是逐结节汇总的。

用法：
    python scripts\\analyze_scope.py
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

LEVELS = (1, 2, 3, 4)
LVL = {1: "V>=1", 2: "V>=2", 3: "V>=3", 4: "V>=4"}
ARM = {"tgtA": "A Union", "tgtB": "B Majority", "tgtC": "C Consensus", "tgtD": "D Soft",
       "archR18": "ResNet-18", "archPU": "Plain U-Net"}
LADDER = ["patch128", "win128", "win192", "win256", "win384", "win512", "slice", "volume"]
W = 108


def boot_ci(x, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    b = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(n)])
    return np.percentile(b, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="outputs/analysis_scope/scope_test.csv")
    ap.add_argument("--patch-csv", default="outputs/analysis_test/per_sample_test.csv",
                    help="已有的逐结节 patch 级结果，用于自检。必须逐结节比对，"
                         "不能拿全量汇总值比子集——那样的偏差全是抽样，读不出流水线对错")
    ap.add_argument("--thr-alias", default="",
                    help="给尚未校准阈值的臂指定借用来源，如 tgtBshift=tgtB。"
                         "逗号分隔。仅用于冒烟测试；正式结果必须用该臂自己的验证集校准阈值")
    ap.add_argument("--thr-from", default="outputs/analysis_full/symmetric_analysis.json",  # 2026-10-02: 旧默认 outputs/analysis/ 是 08-31 的过期副本（实例级校准阈值）
                    help="各臂各层级的校准阈值来源（验证集选定）")
    ap.add_argument("--out", default="outputs/analysis_scope/scope_analysis.json")
    args = ap.parse_args()

    df = pd.read_csv(args.scope)
    df["arm"] = df.run.str.split("_fold").str[0]
    seen = set(df.arm)
    # 已知臂按 ARM 的既定顺序排在前，未知臂（如新增的 tgtBshift）按字典序附后，
    # 不能像之前那样被静默丢掉
    arms = [a for a in ARM if a in seen] + sorted(seen - set(ARM))
    present = [s for s in LADDER if s in set(df.scope)]
    thr_src = json.load(open(args.thr_from, encoding="utf-8"))["table1"]
    alias = dict(kv.split("=", 1) for kv in args.thr_alias.split(",") if "=" in kv)
    missing = [a for a in arms if a not in thr_src and a not in alias]
    if missing:
        raise SystemExit(
            f"这些臂在 {args.thr_from} 里没有验证集校准阈值: {missing}\n"
            f"正式结果的做法：先对该臂跑 eval_symmetric.py（验证折）再跑 analyze_symmetric.py，\n"
            f"让它自己的阈值进入 symmetric_analysis.json。\n"
            f"仅冒烟测试可临时借用：--thr-alias {missing[0]}=tgtB（借来的阈值不得写进论文）")
    if alias:
        print(f"[警告] 借用阈值 {alias} —— 冒烟测试可用，正式结果不可用\n")

    ref = pd.read_csv(args.patch_csv) if os.path.exists(args.patch_csv) else None

    n_cases = df.case_id.nunique()
    n_sl = int(df.groupby("case_id").n_slices.first().sum())
    n_nod = int(df.groupby("case_id").n_nodules.first().sum()) if "n_nodules" in df else -1
    print("=" * W)
    print(f"评测空间 F3 分析  |  病例 {n_cases}  切片 {n_sl}  结节 {n_nod}  臂 {arms}")
    print(f"阶梯: {' -> '.join(present)}")
    print("=" * W)
    rep = {"n_cases": int(n_cases), "n_slices": n_sl, "n_nodules": n_nod,
           "arms": arms, "scopes": present}

    def pick(arm, scope, k):
        """该臂在该层级的【验证集校准阈值】下、该评测空间的逐病例 Dice 与权重。

        【必须先在折上聚合】同一个臂有 5 个折模型，它们跑的是【同一批测试病例】。
        若把 158 病例 x 5 折当成 790 个独立观测直接做配对检验，n 被虚增 5 倍，
        置信区间收窄、p 值虚小——这是把模型间的相关性当成了额外样本量。
        正确的分析单元是病例：每个病例先在 5 折上取均值，再做病例间配对，n = 158。
        """
        t = float(thr_src[alias.get(arm, arm)][str(k)]["thr"])
        s = df[(df.arm == arm) & (df.scope == scope) & (df.level == k)]
        if s.empty:
            return None
        avail = np.sort(s.threshold.unique())
        tt = float(avail[np.argmin(np.abs(avail - t))])
        s = s[np.isclose(s.threshold, tt)]
        g = s.groupby("case_id").agg(dice=("dice", "mean"), n_units=("n_units", "mean"))
        return g.dice, g.n_units, tt

    def best(arm, scope, k):
        """该评测空间自己的最优阈值（重新标定也救不回来时用来证明这一点）。"""
        s = df[(df.arm == arm) & (df.scope == scope) & (df.level == k)]
        if s.empty:
            return None
        # 先按 (阈值, 病例) 在折上取均值，再对病例取均值；直接 groupby(阈值) 会
        # 按 折x病例 的行数加权，折数不齐时给出的不是病例级均值
        g = s.groupby(["threshold", "case_id"]).dice.mean().groupby("threshold").mean()
        return float(g.max()), float(g.idxmax())

    # ---------- 表 0 端到端自检 ----------
    print("\n表 0  自检：patch128 应当复现已有的 patch 级结果")
    print("     （同一模型、同一阈值、同一裁剪；参考值【限定在本次跑到的同一批病例上】重算，")
    print("       否则子集 vs 全量的抽样差会被误读成流水线错误——之前那个 13.47 就是这么来的）")
    print("-" * W)
    print(f"{'run':<20}{'层级':<8}{'参考(同 run 同病例)':>16}{'本脚本 patch128':>18}"
          f"{'差值 x100':>11}{'结节数 本/参':>10}")
    t0 = []
    cases_here = set(df.case_id.unique())
    # 【按 run 比，不能按臂比】per_sample_test.csv 里同一个臂的 5 个折模型都在测试集上
    # 评过一遍，按臂筛会把五折平均当成参考值（n_ref 会是 mine 的 5 倍），
    # 得到的偏差是"单折 vs 五折均值"，与流水线正确性无关。
    for run in sorted(df.run.unique()):
        a_ = run.split("_fold")[0]
        for k in LEVELS:
            t = float(thr_src[alias.get(a_, a_)][str(k)]["thr"])
            s_ = df[(df.run == run) & (df.scope == "patch128") & (df.level == k)]
            if s_.empty:
                continue
            avail = np.sort(s_.threshold.unique())
            tt = float(avail[np.argmin(np.abs(avail - t))])
            s_ = s_[np.isclose(s_.threshold, tt)]
            wm = float((s_.dice * s_.n_units).sum() / s_.n_units.sum())   # 逐结节加权
            rv, n_ref = None, 0
            if ref is not None:
                q = ref[(ref.run == run) & ref.case_id.isin(cases_here)
                        & np.isclose(ref.threshold, tt) & (ref[f"gt_area_v{k}"] > 0)]
                if len(q):
                    rv, n_ref = float(q[f"dice_v{k}"].mean()), len(q)
            gap = None if rv is None else (wm - rv) * 100
            n_mine = int(s_.n_units.sum())
            t0.append(dict(run=run, arm=a_, level=k, thr=tt, ref=rv, n_ref=n_ref,
                           n_mine=n_mine, weighted=wm,
                           gap_pts=None if gap is None else float(gap)))
            flag = "" if (rv is None or n_ref == n_mine) else "  <== 结节数对不上"
            print(f"{run:<20}{LVL[k]:<8}"
                  f"{(f'{rv:.4f}' if rv is not None else '  -  '):>16}{wm:>18.4f}"
                  f"{(f'{gap:+.4f}' if gap is not None else '  -  '):>11}"
                  f"{f'{n_mine}/{n_ref}':>10}{flag}")
    rep["table0_selfcheck"] = t0
    gaps = [abs(x["gap_pts"]) for x in t0 if x.get("gap_pts") is not None]
    print("-" * W)
    if gaps:
        mx = max(gaps)
        # 同一批病例、同一批结节、同一阈值下应当逐位相同，只剩 float32 舍入
        print(f"  最大偏差 {mx:.4f} Dice 点  ->  "
              + ("通过（float32 舍入量级）" if mx < 0.01 else
                 "!! 不一致，先查流水线，不要往下解释 !!"))
        rep["selfcheck_max_gap_pts"] = float(mx)
    else:
        print("  [跳过] 没有可比对的参考。新臂（如平移臂）尚未跑过 eval_symmetric，")
        print("         属正常；已有臂若也无参考，检查 --patch-csv 路径。")

    # ---------- 表 1 完整阶梯 ----------
    print("\n" + "=" * W)
    print("表 1  同一模型、同一阈值，只改评测空间（各层级用验证集选定的阈值）")
    print("=" * W)
    hdr = f"{'臂':<14}{'层级':<8}" + "".join(f"{s:>10}" for s in present)
    print(hdr)
    t1 = {}
    for a in arms:
        for k in LEVELS:
            row = {}
            for sc in present:
                r = pick(a, sc, k)
                if r is not None:
                    row[sc] = float(r[0].mean())
            if not row:
                continue
            t1[f"{a}_v{k}"] = row
            print(f"{ARM.get(a,a):<14}{LVL[k]:<8}" +
                  "".join(f"{row.get(s, np.nan):>10.4f}" for s in present))
    rep["table1_ladder"] = t1

    # ---------- 表 2 逐段归因 ----------
    print("\n" + "=" * W)
    print("表 2  逐段归因：每一段只动一个变量")
    print("=" * W)
    segs = [("patch128", "win128", "输入上下文"),
            ("win128", "win512", "纯视野效应"),
            ("win512", "slice", "窗口->整片"),
            ("slice", "volume", "纳入无结节切片")]
    print(f"{'臂':<14}{'层级':<8}" + "".join(f"{n:>16}" for _, _, n in segs))
    t2 = []
    for a in arms:
        for k in LEVELS:
            e = t1.get(f"{a}_v{k}")
            if not e:
                continue
            vals = []
            for lo, hi, name in segs:
                v = (e[hi] - e[lo]) * 100 if lo in e and hi in e else np.nan
                vals.append(v)
                t2.append(dict(arm=a, level=k, segment=name, delta_pts=None if np.isnan(v) else float(v)))
            print(f"{ARM.get(a,a):<14}{LVL[k]:<8}" +
                  "".join(f"{v:>+16.2f}" for v in vals))
    rep["table2_segments"] = t2
    pure = [x["delta_pts"] for x in t2 if x["segment"] == "纯视野效应" and x["delta_pts"] is not None]
    if pure:
        print("-" * W)
        print(f"  纯视野效应（win128 -> win512）幅度: 最大 {max(abs(v) for v in pure):.2f}，"
              f"平均 {np.mean([abs(v) for v in pure]):.2f} Dice 点")
        rep["pure_fov_max_pts"] = float(max(abs(v) for v in pure))
        rep["pure_fov_mean_pts"] = float(np.mean([abs(v) for v in pure]))

    # ---------- 表 3 重新标定阈值是否能救回来 ----------
    print("\n" + "=" * W)
    print("表 3  每个评测空间用它自己的最优阈值（检验落差是否只是阈值没标定好）")
    print("=" * W)
    print(f"{'臂':<14}{'层级':<8}" + "".join(f"{s:>18}" for s in present))
    t3 = {}
    for a in arms:
        for k in LEVELS:
            cells = []
            for sc in present:
                b = best(a, sc, k)
                cells.append(f"{b[0]:.4f}@{b[1]:g}" if b else "-")
                if b:
                    t3[f"{a}_v{k}_{sc}"] = {"dice_opt": b[0], "thr": b[1]}
            print(f"{ARM.get(a,a):<14}{LVL[k]:<8}" + "".join(f"{c:>18}" for c in cells))
    rep["table3_per_scope_opt"] = t3

    # ---------- 表 4 逐病例配对检验 ----------
    print("\n" + "=" * W)
    print("表 4  逐病例配对检验（同一模型、同一阈值，n = 病例数）")
    print("=" * W)
    print(f"{'臂':<14}{'层级':<8}{'对比':<22}{'差值':>9}{'95%CI':>20}{'p':>12}{'n(病例)':>9}")
    t4 = []
    for a in arms:
        for k in LEVELS:
            for lo, hi, name in segs:
                rl, rh = pick(a, lo, k), pick(a, hi, k)
                if rl is None or rh is None:
                    continue
                x, y = rl[0].align(rh[0], join="inner")
                d = (y - x).values * 100
                if len(d) < 3 or np.allclose(d, 0):
                    continue
                lo_ci, hi_ci = boot_ci(d)
                p = float(stats.ttest_rel(y, x).pvalue)
                t4.append(dict(arm=a, level=k, segment=name, n=len(d),
                               diff=float(d.mean()), ci=[float(lo_ci), float(hi_ci)], p=p))
                print(f"{ARM.get(a,a):<14}{LVL[k]:<8}{f'{hi} - {lo}':<22}{d.mean():>+9.2f}"
                      f"{f'[{lo_ci:+.2f}, {hi_ci:+.2f}]':>20}{p:>12.2e}{len(d):>9}")
    rep["table4_paired"] = t4

    # ---------- 表 5 放回协议因素排序 ----------
    print("\n" + "=" * W)
    print("表 5  把评测空间放回协议因素的排序里")
    print("=" * W)
    rows = []
    if pure:
        rows.append(("评测视野 128->512 窗口（本节实测，干净）", max(abs(v) for v in pure), "协议"))
    conf = [x["delta_pts"] for x in t2
            if x["segment"] == "纳入无结节切片" and x["delta_pts"] is not None]
    if conf:
        rows.append(("切片->体积（与训练分布混杂，不可单独归因）",
                     max(abs(v) for v in conf), "混杂"))
    rows += [("评测靶 V>=1..4（固定阈值 0.5）", 16.42, "协议"),
             ("二值化阈值（并集臂 @V>=4）", 10.02, "协议"),
             ("训练靶 4 臂极差 @V>=4", 2.99, "方法"),
             ("架构 3 模型极差 @V>=2", 1.43, "模型"),
             ("单次训练随机波动 2sigma", 1.12, "噪声")]
    for name, val, typ in rows:
        print(f"  {name:<46}{typ:>6}{val:>9.2f} 点")
    print("\n  [注] 后五项来自 5 折验证集与种子重复实验（见主分析），此处并列仅为量级对照。")
    print("  [注] 标为'混杂'的一行不能写成评测空间效应：本研究的模型只在以结节为中心的")
    print("       patch 上训练，从未见过不含结节的区域，该落差同时包含训练分布的影响。")
    rep["table5_ranking"] = [{"name": n, "pts": float(v), "type": t} for n, v, t in rows]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    print(f"\n已保存 -> {args.out}")


if __name__ == "__main__":
    main()
