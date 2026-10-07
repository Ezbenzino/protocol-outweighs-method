# -*- coding: utf-8 -*-
"""统计论文 3.1 节需要的数据集数字，并核验划分的正确性。

**这个脚本的第一职责是核验，不是统计。** 起因：项目文档里长期写着
"训练/测试划分 858/202"，实际核对发现那是一个早期的非交叉验证划分
（train.csv + val.csv），它与 test.csv 有 176 个病例重叠。所有交叉验证
实验用的是 train_foldK / val_foldK，与 test.csv 零重叠，是干净的；
但 858 这个数字若被写进论文就是错的。脚本每次运行都重新验证这一点。

输出：
  A. 划分构成与重叠核验（含 train.csv/val.csv 这条历史遗留划分的警示）
  B. 结节数与直径分布（按划分）
  C. 各共识层级的 GT 面积与空掩膜比例（从已有的评测 CSV 读，不重算）
  D. 原始 XML 标注量 vs 最终保留的结节数

用法（PowerShell，在仓库根目录下）：
    python scripts\\dataset_stats.py
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

W = 78


def read_split(path):
    """读划分文件，去掉表头与空行。表头是字面量 case_id。"""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and ln.strip() != "case_id"]


def pct(a, b):
    return f"{100.0 * a / b:.1f}%" if b else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--npz", default="data/processed/npz")
    ap.add_argument("--ann", default="data/processed/annotations.json")
    ap.add_argument("--analysis", default="outputs/analysis")
    ap.add_argument("--analysis-test", default="outputs/analysis_test")
    ap.add_argument("--out", default="outputs/analysis/dataset_stats.json")
    args = ap.parse_args()

    rep = {}

    # ---------------- A. 划分核验 ----------------
    print("=" * W)
    print("A. 划分构成与重叠核验")
    print("=" * W)
    folds = [k for k in range(5) if read_split(os.path.join(args.splits, f"val_fold{k}.csv"))]
    cv_val = {k: read_split(os.path.join(args.splits, f"val_fold{k}.csv")) for k in folds}
    cv_trn = {k: read_split(os.path.join(args.splits, f"train_fold{k}.csv")) for k in folds}
    test = read_split(os.path.join(args.splits, "test.csv")) or []
    cv_pool = sorted(set().union(*cv_val.values())) if cv_val else []

    print(f"  交叉验证池（5 折验证集并集）: {len(cv_pool)} 例")
    for k in folds:
        both = set(cv_trn[k]) & set(cv_val[k])
        print(f"    fold{k}: 训练 {len(cv_trn[k])} + 验证 {len(cv_val[k])} "
              f"= {len(cv_trn[k]) + len(cv_val[k])}   折内重叠 {len(both)}")
    print(f"  保留测试集: {len(test)} 例")
    leak = sorted(set(cv_pool) & set(test))
    print(f"  交叉验证池 与 测试集 重叠: {len(leak)} 例  ->  "
          + ("通过" if not leak else f"!! 泄漏 !! {leak[:5]}"))

    # 各折验证集是否互斥
    dup = [c for c in cv_pool if sum(c in cv_val[k] for k in folds) > 1]
    print(f"  各折验证集互斥性: 重复出现的病例 {len(dup)} 例  ->  "
          + ("通过" if not dup else "!! 不互斥 !!"))

    # 历史遗留划分的警示
    old_tr, old_va = read_split(os.path.join(args.splits, "train.csv")), \
                     read_split(os.path.join(args.splits, "val.csv"))
    if old_tr and old_va:
        old = set(old_tr) | set(old_va)
        ov = sorted(old & set(test))
        print(f"\n  [历史遗留] train.csv + val.csv = {len(old)} 例，"
              f"与测试集重叠 {len(ov)} 例")
        if ov:
            print(f"  !! 该划分已被测试集污染，任何用它得到的结果都不可用。"
                  f"论文中的划分数字必须用交叉验证池 {len(cv_pool)} / 测试集 {len(test)}，"
                  f"不要用 {len(old)}。")
    rep["splits"] = {"cv_pool": len(cv_pool), "test": len(test),
                     "per_fold": {k: {"train": len(cv_trn[k]), "val": len(cv_val[k])} for k in folds},
                     "cv_test_overlap": len(leak), "fold_val_duplicates": len(dup),
                     "legacy_split_size": len(old) if old_tr and old_va else None,
                     "legacy_test_overlap": len(ov) if old_tr and old_va else None}

    # ---------------- B. 结节数与直径 ----------------
    print("\n" + "=" * W)
    print("B. 结节数与直径分布（按最终纳入规则：多数投票连通域 + 等效直径 >= 3 mm）")
    print("=" * W)

    def collect(case_ids):
        dias, n_with, n_missing = [], 0, 0
        for c in case_ids:
            f = os.path.join(args.npz, f"{c}.json")
            if not os.path.exists(f):
                n_missing += 1
                continue
            with open(f, encoding="utf-8") as fh:
                nods = json.load(fh).get("nodules", [])
            if nods:
                n_with += 1
            dias += [float(n["diameter_mm"]) for n in nods]
        return np.array(dias), n_with, n_missing

    rep["nodules"] = {}
    print(f"  {'划分':<14}{'病例':>7}{'有结节':>8}{'结节数':>8}{'直径 mm  中位(IQR)':>26}{'范围':>16}")
    for name, ids in (("交叉验证池", cv_pool), ("测试集", test)):
        d, nw, nm = collect(ids)
        if len(d) == 0:
            continue
        q1, q2, q3 = np.percentile(d, [25, 50, 75])
        print(f"  {name:<14}{len(ids):>7}{nw:>8}{len(d):>8}"
              f"{f'{q2:.1f} ({q1:.1f}-{q3:.1f})':>26}{f'{d.min():.1f}-{d.max():.1f}':>16}")
        bins = [(3, 5), (5, 10), (10, 30), (30, 1e9)]
        names = ["3-5 mm", "5-10 mm", "10-30 mm", ">30 mm"]
        counts = [int(((d >= lo) & (d < hi)).sum()) for lo, hi in bins]
        print(f"  {'':14}尺寸分布: " + "  ".join(
            f"{n}={c} ({pct(c, len(d))})" for n, c in zip(names, counts)))
        if nm:
            print(f"  {'':14}[注] {nm} 个病例缺少元数据文件")
        rep["nodules"][name] = {
            "cases": len(ids), "cases_with_nodules": nw, "n_nodules": int(len(d)),
            "diam_median": float(q2), "diam_q1": float(q1), "diam_q3": float(q3),
            "diam_min": float(d.min()), "diam_max": float(d.max()),
            "size_bins": dict(zip(names, counts)), "missing_meta": nm}

    # ---------------- C. 各共识层级的 GT ----------------
    print("\n" + "=" * W)
    print("C. 各共识层级的 GT 面积与空掩膜比例（取自评测导出的逐样本表）")
    print("=" * W)
    rep["gt_levels"] = {}
    for tag, d in (("验证（5 折）", args.analysis), ("测试集", args.analysis_test)):
        fs = sorted(glob.glob(os.path.join(d, "per_sample_free_*.csv")))
        if not fs:
            continue
        import pandas as pd
        fr = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
        # GT 与实验臂无关，同一批样本被每个臂各写了一遍，必须去重。
        # 按 run 筛会漏折（验证集有 5 折）；按臂筛会在测试集上重复 5 次
        # （测试集里 5 个折模型跑的是同一批样本）。正确做法：每折取一个 run。
        keep = set(fr.groupby("fold").run.first())
        one = fr[fr.run.isin(keep)]
        print(f"\n  [{tag}]  样本数 {len(one)}")
        print(f"    {'层级':<8}{'平均面积 px':>14}{'非空样本':>10}{'非空占比':>10}")
        lv = {}
        for k in (1, 2, 3, 4):
            col = one[f"gt_area_v{k}"]
            lv[k] = {"mean_area": float(col.mean()),
                     "n_nonempty": int((col > 0).sum()), "n_total": int(len(col))}
            print(f"    V>={k:<6}{col.mean():>14.1f}{int((col > 0).sum()):>10}"
                  f"{pct((col > 0).sum(), len(col)):>10}")
        rep["gt_levels"][tag] = lv

    # ---------------- D. 原始标注量 ----------------
    print("\n" + "=" * W)
    print("D. 原始 XML 标注量 vs 最终保留的结节")
    print("=" * W)
    if os.path.exists(args.ann):
        with open(args.ann, encoding="utf-8") as f:
            ann = json.load(f)
        n_series, n_sess, n_raw = len(ann), 0, 0
        for v in ann.values():
            sess = v.get("sessions", [])
            n_sess += len(sess)
            for s in sess:
                n_raw += len(s.get("nodules", []))
        n_kept = sum(r["n_nodules"] for r in rep["nodules"].values())
        print(f"  解析到的序列数: {n_series}")
        print(f"  阅片 session 数: {n_sess}  (每序列平均 {n_sess / max(n_series,1):.2f})")
        print(f"  原始标注对象总数（未去重、未按结节合并）: {n_raw}")
        print(f"  最终纳入本研究的结节-切片样本数: {n_kept}")
        print(f"  [说明] 前者是逐医生、逐 ROI 的原始标注计数，后者是经"
              f"「多数投票连通域 + 直径>=3mm」重定义并落入本研究划分的样本数；"
              f"两者口径不同，不可直接相减解释为「被排除数」。")
        rep["annotations"] = {"n_series": n_series, "n_sessions": n_sess,
                              "n_raw_objects": n_raw, "n_kept_samples": n_kept}
    else:
        print(f"  [跳过] 找不到 {args.ann}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    print(f"\n已保存 -> {args.out}")


if __name__ == "__main__":
    main()
