import json, os, glob
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS = os.path.join(ROOT, 'outputs', 'analysis')


def load(csvs, n):
    """分析单元 = 病例。

    per_sample_*.csv 的每一行是一个结节实例（LIDC 每例中位 9 个、最多 79 个），
    且同一实例在每个 (run, threshold) 组合下重复出现。因此先在每个文件内取
    任一 (run, threshold) 切片，使每个实例恰好出现一次，再按 case_id 汇总
    GT 面积，得到真正的病例级量。

    早先版本用 drop_duplicates('case_id')，只保留每例的第一个实例，
    得到的既不是病例级也不是实例级统计量（见 docs/分析单元审计_20260904.md）。
    """
    cols = [f'gt_area_v{k}' for k in range(1, n + 1)]
    parts = []
    for f in csvs:
        d = pd.read_csv(f)
        r0 = sorted(d.run.unique())[0]
        t0 = sorted(d.threshold.unique())[0]
        parts.append(d[(d.run == r0) & (d.threshold == t0)][['case_id'] + cols])
    inst = pd.concat(parts, ignore_index=True)
    return inst.groupby('case_id', as_index=False)[cols].sum()


def disagreement(sub, n):
    v1 = sub[f'gt_area_v1']; vN = sub[f'gt_area_v{n}']
    ratio = vN / v1.replace(0, pd.NA)
    return {'area_mean_ratio': round(float(vN.sum()/v1.sum()), 4),
            'per_case_mean': round(float(ratio.dropna().mean()), 4),
            'no_consensus_frac': round(float((vN == 0).mean()), 4),
            'n_cases': len(sub)}


# 数据集定义：(glob 模式, n_raters, 标签)
# LIDC 用主分析 per_sample（病例级口径），QUBIQ 各任务独立目录。
DATASET_SPECS = [
    ('LIDC', sorted(glob.glob(r'outputs/analysis_full/per_sample_val_fold*.csv'))
     + [r'outputs/analysis_test/per_sample_test.csv'], 4),
    ('QUBIQ-brain', sorted(glob.glob(r'outputs/analysis/qubiq_brain/per_sample_val_fold*.csv')), 7),
    ('QUBIQ-kidney', sorted(glob.glob(r'outputs/analysis/qubiq_kidney/per_sample_val_fold*.csv')), 3),
    ('QUBIQ-brain-tumor-task1', sorted(glob.glob(r'outputs/analysis/qubiq_brain_tumor_task01/per_sample_val_fold*.csv')), 3),
    ('QUBIQ-brain-tumor-task2', sorted(glob.glob(r'outputs/analysis/qubiq_brain_tumor_task02/per_sample_val_fold*.csv')), 3),
    ('QUBIQ-brain-tumor-task3', sorted(glob.glob(r'outputs/analysis/qubiq_brain_tumor_task03/per_sample_val_fold*.csv')), 3),
    ('QUBIQ-prostate-task1', sorted(glob.glob(r'outputs/analysis/qubiq_prostate_task01/per_sample_val_fold*.csv')), 6),
    ('QUBIQ-prostate-task2', sorted(glob.glob(r'outputs/analysis/qubiq_prostate_task02/per_sample_val_fold*.csv')), 6),
    ('QUBIQ-pancreas', sorted(glob.glob(r'outputs/analysis/qubiq_pancreas/per_sample_val_fold*.csv')), 2),
    ('QUBIQ-pancreatic-lesion', sorted(glob.glob(r'outputs/analysis/qubiq_pancreatic_lesion/per_sample_val_fold*.csv')), 2),
]

res = {
    'definition': '面积均值之比 = Σ|V>=N| / Σ|V>=1| (case-level GT areas), 同代码跨数据集',
}
for label, csvs, n in DATASET_SPECS:
    if not csvs:
        print(f'[缺失] {label}: 无 per_sample CSV（训练未完成）')
        continue
    sub = load(csvs, n)
    res[label] = disagreement(sub, n)
    print(f"{label:<26} {res[label]['area_mean_ratio']:.4f}  n_cases={res[label]['n_cases']}")

json.dump(res, open(os.path.join(ANALYSIS, 'disagreement_index.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2)
print('\n已写: outputs/analysis/disagreement_index.json')
