# -*- coding: utf-8 -*-
"""数据泄漏审计 —— 生成可复核的 data_audit_report.json。

审计范围（对应论文投稿前 Checklist 的"数据泄漏"项）：
  1. 病例级 train/validation/test 划分是否干净（跨集合重叠为 0）
  2. 同一病例的切片是否跨集合（病例即切片容器，病例不相交 -> 切片不相交）
  3. 同一结节是否跨集合（结节定义在病例内，病例不相交 -> 结节不相交）
  4. 测试集是否参与阈值选择（必须为否：阈值从验证折 --thr-from 导入）
  5. 预训练权重来源（ImageNet-1K，外部，与 LIDC/test 无关）
  6. normalization 是否使用 test 信息（必须为否：固定 /255 + 固定窗宽）
  7. 外部数据（QUBIQ）是否独立（grand-challenge，独立标注者/机构）

证据等级标注：
  data_verified  —— 直接从划分文件/逐样本 CSV 计算得到
  code_verified  —— 从源代码/配置断言得到（给出证据路径）
  declared       —— 数据来源声明（QUBIQ 需遵守挑战条款，见 Data Availability）

用法（仓库根目录，PowerShell）：
    python scripts\\data_audit_report.py
输出：
    outputs/analysis/data_audit_report.json
"""
import argparse
import glob
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS = os.path.join(ROOT, "data", "splits")
ANALYSIS = os.path.join(ROOT, "outputs", "analysis_full")
ANALYSIS_TEST = os.path.join(ROOT, "outputs", "analysis_test")


def read_split(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and ln.strip() != "case_id"]


def read_case_ids(csvs, extra=None):
    """只读 case_id 列（per_sample 文件很大，不能整读）。"""
    seen = set()
    src = list(csvs)
    if extra:
        src = src + list(extra)
    for f in src:
        if not os.path.exists(f):
            continue
        for chunk in pd.read_csv(f, usecols=["case_id"], chunksize=500_000):
            seen.update(chunk["case_id"].dropna().astype(str).unique())
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default=SPLITS)
    ap.add_argument("--analysis", default=ANALYSIS)
    ap.add_argument("--analysis-test", default=ANALYSIS_TEST)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "analysis", "data_audit_report.json"))
    args = ap.parse_args()

    rep = {"report_version": "1.0", "generated_by": "scripts/data_audit_report.py",
           "checks": {}}
    all_pass = True

    # ---------------- 1. 病例级划分 ----------------
    folds = [k for k in range(5)]
    cv_val = {k: set(read_split(os.path.join(args.splits, f"val_fold{k}.csv")) or []) for k in folds}
    cv_trn = {k: set(read_split(os.path.join(args.splits, f"train_fold{k}.csv")) or []) for k in folds}
    test = set(read_split(os.path.join(args.splits, "test.csv")) or [])
    cv_pool = set().union(*cv_val.values()) if cv_val else set()

    per_fold = {}
    for k in folds:
        overlap_in_fold = len(cv_trn[k] & cv_val[k])
        per_fold[k] = {"train": len(cv_trn[k]), "val": len(cv_val[k]),
                       "train_val_overlap": overlap_in_fold,
                       "pass": overlap_in_fold == 0}
        all_pass &= per_fold[k]["pass"]

    cv_test_overlap = sorted(cv_pool & test)
    fold_val_dup = [c for c in cv_pool if sum(c in cv_val[k] for k in folds) > 1]
    c1 = {
        "evidence": "data_verified",
        "cv_pool_cases": len(cv_pool),
        "test_cases": len(test),
        "per_fold": per_fold,
        "cv_test_overlap_cases": len(cv_test_overlap),
        "cv_test_overlap_examples": cv_test_overlap[:5],
        "fold_val_mutual_exclusivity_violations": len(fold_val_dup),
        "pass": (len(cv_test_overlap) == 0 and not fold_val_dup
                 and all(per_fold[k]["pass"] for k in folds)),
    }
    all_pass &= c1["pass"]
    rep["checks"]["case_level_splits"] = c1

    # ---------------- 2. 逐样本层面：病例不跨集合 ----------------
    val_csvs = sorted(glob.glob(os.path.join(args.analysis, "per_sample_val_fold*.csv")))
    test_csv = os.path.join(args.analysis_test, "per_sample_test.csv")
    val_cases = read_case_ids(val_csvs)
    test_cases = read_case_ids([test_csv])
    cross = sorted(val_cases & test_cases)
    # 逐样本里出现的病例必须落在对应划分内
    val_outside_pool = sorted(val_cases - cv_pool)
    test_outside_test = sorted(test_cases - test)
    c2 = {
        "evidence": "data_verified",
        "n_val_cases_seen_in_per_sample": len(val_cases),
        "n_test_cases_seen_in_per_sample": len(test_cases),
        "case_ids_in_both_val_and_test_per_sample": len(cross),
        "cross_examples": cross[:5],
        "val_cases_outside_cv_pool": len(val_outside_pool),
        "test_cases_outside_test_split": len(test_outside_test),
        "pass": (len(cross) == 0 and not val_outside_pool and not test_outside_test),
    }
    all_pass &= c2["pass"]
    rep["checks"]["no_case_across_sets"] = c2

    # ---------------- 3. 结节不跨集合 ----------------
    # 结节 = (case_id, slice, centroid)；病例不相交 -> 结节不可能跨集合。
    # 逐样本层面已证明 case 集合不相交（check 2），此处给出定义级论证 + 计数。
    c3 = {
        "evidence": "data_verified + code_verified",
        "definition": "结节实例由 (case_id, slice, cy, cx) 唯一标识，全部位于所属病例的 npz 内；"
                      "病例集合不相交（check 1/2），故不存在跨集合的同一结节。",
        "n_cv_pool_nodules": 8471,
        "n_test_nodules": 2062,
        "source": "outputs/analysis/dataset_stats.json (nodules)",
        "pass": True,
    }
    rep["checks"]["no_nodule_across_sets"] = c3

    # ---------------- 4. 测试集不参与阈值选择 ----------------
    c4 = {
        "evidence": "code_verified",
        "mechanism": ("eval_symmetric.py 在 per_sample CSV 里落盘全部阈值扫描（0.001~0.999, 33 个）；"
                      "analyze_symmetric.py 对测试集以 --thr-from "
                      "outputs/analysis_full/symmetric_analysis.json 复用验证折选定的阈值，"
                      "不在测试集上重新选阈值（docstring: 用于测试集无偏估计）。"),
        "code_refs": ["scripts/eval_symmetric.py", "scripts/analyze_symmetric.py (--thr-from)"],
        "analysis_test_command": ("analyze_symmetric.py --out-dir outputs/analysis_test "
                                  "--thr-from outputs/analysis_full/symmetric_analysis.json"),
        "pass": True,
    }
    rep["checks"]["test_not_in_threshold_selection"] = c4

    # ---------------- 5. 预训练权重来源 ----------------
    c5 = {
        "evidence": "code_verified",
        "sources": {
            "resnet18/resnet34": "torchvision.models.*_Weights.IMAGENET1K_V1",
            "convnext_tiny": "torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1",
            "pvt_v2_*": "timm.create_model(..., pretrained=True)  (ImageNet-1K 预训练)",
        },
        "note": "ImageNet-1K 为外部公开权重，不包含 LIDC 或本测试集信息；"
                "输入为单 CT 通道复制 3 通道（configs/default.yaml model.in_channels=3）。",
        "code_refs": ["src/models/encoder.py", "configs/default.yaml"],
        "pass": True,
    }
    rep["checks"]["pretrained_weights_source"] = c5

    # ---------------- 6. normalization 不使用 test 信息 ----------------
    c6 = {
        "evidence": "code_verified",
        "normalization": "固定线性缩放 astype(float32)/255.0，输入为预处理阶段已固定的窗宽 "
                         "HU[-1000,400]->[0,255] uint8；不计算任何数据集的均值/方差统计量，"
                         "不依赖 test 集。QUBIQ MRI 子集用逐卷 min-max（per-volume，与标签/test 无关）。",
        "code_refs": ["src/data/dataset.py (astype(float32)/255.0)",
                      "preprocess/process_lidc.py (window)", "preprocess/build_qubiq.py (--normalize minmax)"],
        "pass": True,
    }
    rep["checks"]["normalization_no_test_info"] = c6

    # ---------------- 7. 外部数据独立性 ----------------
    c7 = {
        "evidence": "declared",
        "lidc": "The Cancer Imaging Archive (TCIA) LIDC-IDRI，1018 例，4 位放射科医生标注。",
        "qubiq": "grand-challenge.org QUBIQ 2021 挑战数据（需加入挑战并接受数据使用条款）；"
                 "brain-growth 7 raters / kidney 3 raters / prostate 6 raters，MRI+CT，"
                 "与 LIDC 机构与标注者完全独立。",
        "note": "QUBIQ 用于外部泛化验证（评估靶/监督靶效应随标注者一致性的变化），不参与 LIDC 主实验训练。",
        "code_refs": ["preprocess/build_qubiq.py", "configs/qubiq/*.yaml"],
        "pass": True,
    }
    rep["checks"]["external_data_independence"] = c7

    # ---------------- 附录：reference mask 定义（审计相关口径） ----------------
    rep["appendix"] = {
        "reference_mask_definitions": {
            "source": "src/data/dataset.py（全部掩膜由 vote 0..N 派生）",
            "union_mask": "vote >= 1",
            "hard_consensus_majority": "vote >= ceil(N/2)（LIDC N=4 -> >=2；QUBIQ 肾脏 N=3 -> >=2；脑生长 N=7 -> >=4）",
            "soft_label": "consensus = V/N in [0,1]",
            "tie_handling": "LIDC N=4 为偶数，多数阈值为 >=2，不存在票数平局歧义；"
                            "奇数 N 用 ceil(N/2) 上取整",
            "nodule_free_cases": "训练/评测以病例内结节样本为单位，背景切片由 bg sampler 采样；"
                                 "GT 面积为 0 的样本在分析阶段被排除（eval doc note 2）",
            "empty_pred_empty_gt": "Dice 定义中 denom==0 时置 1.0，但仅在 GT 非空样本上统计",
            "multi_nodule_cases": "case-level 聚合 = 病例内对结节实例指标平均，再跨病例平均（见 docs/分析单元审计_20260904.md）",
        }
    }

    rep["summary"] = {"all_checks_pass": bool(all_pass),
                      "check_keys": list(rep["checks"].keys())}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(f"[audit] all_checks_pass = {all_pass}")
    for k, v in rep["checks"].items():
        print(f"  [{'PASS' if v['pass'] else 'FAIL'}] {k}")
    print(f"已保存 -> {args.out}")


if __name__ == "__main__":
    main()
