# QUBIQ 外部验证分析 v2 —— 支持全部 9 个任务点（病例级口径）
# 输入: outputs/analysis/qubiq_{dataset}/per_sample_val_fold{0-4}.csv
# 输出: outputs/analysis/qubiq_all_effects.json
#   protocol  = 评测靶(协议)效应 = max over arms of |Dice@V>=1 - Dice@V>=N|（病例级）
#   supervision = 训练靶效应 = max over targets of (max arm - min arm)
#   两口径：fixed0.5 / opt_thr（与 make_table_generalisation.py 完全一致）
import glob, os, json
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS = os.path.join(ROOT, 'outputs', 'analysis')

# 数据集定义：(目录, n_raters, 前缀列表, 展示名, 模态)
DATASETS = [
    dict(dir=os.path.join(ANALYSIS, 'qubiq_brain'), n=7,
         prefixes=['qubiq_brain_A', 'qubiq_brain_B', 'qubiq_brain_C'],
         label='QUBIQ-brain-growth', modality='MRI'),
    dict(dir=os.path.join(ANALYSIS, 'qubiq_kidney'), n=3,
         prefixes=['qubiq_kidney_A', 'qubiq_kidney_B', 'qubiq_kidney_C'],
         label='QUBIQ-kidney', modality='CT'),
    dict(dir=os.path.join(ANALYSIS, 'qubiq_brain_tumor_task01'), n=3,
         prefixes=['qubiq_brain_tumor_task01_A', 'qubiq_brain_tumor_task01_B', 'qubiq_brain_tumor_task01_C'],
         label='QUBIQ-brain-tumor-task1', modality='MRI'),
    dict(dir=os.path.join(ANALYSIS, 'qubiq_brain_tumor_task02'), n=3,
         prefixes=['qubiq_brain_tumor_task02_A', 'qubiq_brain_tumor_task02_B', 'qubiq_brain_tumor_task02_C'],
         label='QUBIQ-brain-tumor-task2', modality='MRI'),
    dict(dir=os.path.join(ANALYSIS, 'qubiq_brain_tumor_task03'), n=3,
         prefixes=['qubiq_brain_tumor_task03_A', 'qubiq_brain_tumor_task03_B', 'qubiq_brain_tumor_task03_C'],
         label='QUBIQ-brain-tumor-task3', modality='MRI'),
    dict(dir=os.path.join(ANALYSIS, 'qubiq_prostate_task01'), n=6,
         prefixes=['qubiq_prostate_task01_A', 'qubiq_prostate_task01_B', 'qubiq_prostate_task01_C'],
         label='QUBIQ-prostate-task1', modality='MRI'),
    dict(dir=os.path.join(ANALYSIS, 'qubiq_prostate_task02'), n=6,
         prefixes=['qubiq_prostate_task02_A', 'qubiq_prostate_task02_B', 'qubiq_prostate_task02_C'],
         label='QUBIQ-prostate-task2', modality='MRI'),
    dict(dir=os.path.join(ANALYSIS, 'qubiq_pancreas'), n=2,
         prefixes=['qubiq_pancreas_A', 'qubiq_pancreas_B', 'qubiq_pancreas_C'],
         label='QUBIQ-pancreas', modality='CT'),
    dict(dir=os.path.join(ANALYSIS, 'qubiq_pancreatic_lesion'), n=2,
         prefixes=['qubiq_pancreatic_lesion_A', 'qubiq_pancreatic_lesion_B', 'qubiq_pancreatic_lesion_C'],
         label='QUBIQ-pancreatic-lesion', modality='CT'),
]


def load_table(cfg):
    fps = sorted(glob.glob(os.path.join(cfg['dir'], 'per_sample_val_fold*.csv')))
    if not fps:
        return None
    df = pd.concat([pd.read_csv(f) for f in fps], ignore_index=True)
    n = cfg['n']
    cols = [f'dice_v{k}' for k in range(1, n + 1)]
    # 分析单元 = 病例：先在病例内对实例平均，再跨病例平均（与 LIDC 侧同口径）
    agg = (df.groupby(['run', 'threshold', 'case_id'])[cols].mean()
             .groupby(['run', 'threshold']).mean().reset_index())
    return agg


def arm_summary(agg, run_prefix, n, thr_mode):
    sub = agg[agg['run'].str.startswith(run_prefix)]
    out = {}
    for k in range(1, n + 1):
        col = f'dice_v{k}'
        if thr_mode == 'opt':
            out[k] = sub.loc[sub.groupby('run')[col].idxmax(), col].mean()
        else:
            out[k] = sub.loc[sub['threshold'] == 0.5, col].mean()
    return out


def dataset_effects(cfg):
    agg = load_table(cfg)
    if agg is None:
        return None
    n = cfg['n']
    res = {}
    for mode in ['fixed0.5', 'opt_thr']:
        mm = '05' if mode == 'fixed0.5' else 'opt'
        sums = {p: arm_summary(agg, p, n, mm) for p in cfg['prefixes']}
        proto = max(abs(sums[p][1] - sums[p][n]) for p in cfg['prefixes'])
        tgt = max(max(sums[p][k] for p in cfg['prefixes']) - min(sums[p][k] for p in cfg['prefixes'])
                  for k in range(1, n + 1))
        res[mode] = {'protocol': proto * 100, 'supervision': tgt * 100,
                     'ratio': proto / tgt if tgt else float('nan')}
    return res


def main():
    out = {}
    for cfg in DATASETS:
        e = dataset_effects(cfg)
        if e is None:
            print(f"[缺失] {cfg['label']}: 无 per_sample CSV（训练未完成）")
            continue
        out[cfg['label']] = {'n_raters': cfg['n'], 'modality': cfg['modality'],
                             'effects': e}
        print(f"### {cfg['label']} (N={cfg['n']}, {cfg['modality']})")
        for m in ['fixed0.5', 'opt_thr']:
            print(f"  [{m}] protocol={e[m]['protocol']:.2f} supervision={e[m]['supervision']:.2f} "
                  f"ratio={e[m]['ratio']:.2f}x")
    with open(os.path.join(ANALYSIS, 'qubiq_all_effects.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n已写: {os.path.join(ANALYSIS, 'qubiq_all_effects.json')}")


if __name__ == '__main__':
    main()
