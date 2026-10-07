# 论文 Table：Generalisation across datasets（统一口径，可复现）
# 定义（与 analyze_qubiq.py 对 QUBIQ 完全一致）：
#   协议(评测靶)效应 = max over 训练臂 of |Dice@V>=1_opt - Dice@V>=N_opt|
#   训练靶效应       = max over 评测靶 of (max arm - min arm)，每臂各取最优阈值
# 两口径：固定0.5（所有臂所有靶同一阈值）/ 逐靶标定最优阈值
# 输出：markdown 表格 + JSON（供论文引用）
import json, glob, os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS_DIR = os.path.join(ROOT, 'outputs', 'analysis')
ANALYSIS_FULL_DIR = os.path.join(ROOT, 'outputs', 'analysis_full')  # 主分析（21 臂）
# 注意：outputs/analysis/symmetric_analysis.json 是种子噪声分析，勿用作主表数据源

def load_lidc():
    with open(os.path.join(ANALYSIS_FULL_DIR, 'symmetric_analysis.json'), encoding='utf-8') as f:
        lidc = json.load(f)
    return lidc['table1']

def lidc_effects(t1, arms):
    def span_arm(arm, key):
        return abs(t1[arm]['1'][key] - t1[arm]['4'][key])
    def tgt_span(key):
        spans = []
        for k in ['1', '2', '3', '4']:
            vs = [t1[a][k][key] for a in arms]
            spans.append(max(vs) - min(vs))
        return max(spans)
    out = {}
    for mode, key in [('fixed0.5', 'dice_05'), ('opt_thr', 'dice_opt')]:
        proto = max(span_arm(a, key) for a in arms)
        tgt = tgt_span(key)
        out[mode] = {'protocol': proto * 100, 'supervision': tgt * 100,
                     'ratio': proto / tgt if tgt else float('nan')}
    return out

def qubiq_effects(cfg):
    rows = []
    for fp in sorted(glob.glob(os.path.join(cfg['dir'], 'per_sample_val_fold*.csv'))):
        rows.append(pd.read_csv(fp))
    df = pd.concat(rows, ignore_index=True)
    n = cfg['n']
    cols = [f'dice_v{k}' for k in range(1, n + 1)]
    # 分析单元 = 病例：先在病例内对实例平均，再跨病例平均（与 LIDC 侧同口径）
    agg = (df.groupby(['run', 'threshold', 'case_id'])[cols].mean()
             .groupby(['run', 'threshold']).mean().reset_index())
    prefixes = cfg['arms']
    def arm_summary(prefix, mode):
        sub = agg[agg['run'].str.startswith(prefix)]
        out = {}
        for k in range(1, n + 1):
            col = f'dice_v{k}'
            if mode == 'opt':
                out[k] = sub.loc[sub.groupby('run')[col].idxmax(), col].mean()
            else:
                out[k] = sub.loc[sub['threshold'] == 0.5, col].mean()
        return out
    res = {}
    for mode in ['fixed0.5', 'opt_thr']:
        mm = '05' if mode == 'fixed0.5' else 'opt'
        sums = {p: arm_summary(p, mm) for p in prefixes}
        proto = max(abs(sums[p][1] - sums[p][n]) for p in prefixes)
        tgt = max(max(sums[p][k] for p in prefixes) - min(sums[p][k] for p in prefixes)
                  for k in range(1, n + 1))
        res[mode] = {'protocol': proto * 100, 'supervision': tgt * 100,
                     'ratio': proto / tgt if tgt else float('nan')}
    return res

KID = dict(dir=os.path.join(ANALYSIS_DIR, 'qubiq_kidney'), n=3,
           arms=['qubiq_kidney_A', 'qubiq_kidney_B', 'qubiq_kidney_C'])
BRA = dict(dir=os.path.join(ANALYSIS_DIR, 'qubiq_brain'), n=7,
           arms=['qubiq_brain_A', 'qubiq_brain_B', 'qubiq_brain_C'])

t1 = load_lidc()
LIDC_4 = lidc_effects(t1, ['tgtA', 'tgtB', 'tgtC', 'tgtD'])
LIDC_5 = lidc_effects(t1, ['tgtA', 'tgtB', 'tgtC', 'tgtD', 'tgtE'])
BRA_E = qubiq_effects(BRA)
KID_E = qubiq_effects(KID)

print('#' * 72)
print('# Table: Generalisation across datasets and rater disagreement (Dice points)')
print('#' * 72)
print()
print('| Dataset | raters | disagg. idx | mode | protocol | supervision | ratio |')
print('|---|---|---|---|---|---|---|')
rows = {}
with open(os.path.join(ANALYSIS_DIR, 'disagreement_index.json'), encoding='utf-8') as f:
    _DIS = json.load(f)
_IDX = {'LIDC': _DIS['LIDC']['area_mean_ratio'],
        'QUBIQ-brain': _DIS['QUBIQ-brain']['area_mean_ratio'],
        'QUBIQ-kidney': _DIS['QUBIQ-kidney']['area_mean_ratio']}
for label, eff, r, di in [('LIDC', LIDC_5, 4, _IDX['LIDC']),
                          ('QUBIQ-brain', BRA_E, 7, _IDX['QUBIQ-brain']),
                          ('QUBIQ-kidney', KID_E, 3, _IDX['QUBIQ-kidney'])]:
    rows[label] = {}
    for mode in ['fixed0.5', 'opt_thr']:
        e = eff[mode]
        rows[label][mode] = e
        mode_label = 'fixed 0.5' if mode == 'fixed0.5' else 'opt. thr.'
        print(f'| {label} | {r} | {di} | {mode_label} | {e["protocol"]:.2f} | '
              f'{e["supervision"]:.2f} | {e["ratio"]:.2f}x |')
print()
print("注: protocol = 评测靶效应（max over arms of |Dice@V>=1 - Dice@V>=N| at that arm's threshold）")
print('    supervision = 训练靶效应（max over evaluation targets of between-arm spread）')
print('    臂数: LIDC 4 (tgtA-D) + tgtE 一致; QUBIQ 3 (union/majority/consensus). 比值=protocol/supervision')
print()
print('=== 最优阈值口径（论文 5.5 主口径）===')
for label in ['LIDC', 'QUBIQ-brain', 'QUBIQ-kidney']:
    e = rows[label]['opt_thr']
    print(f'{label}: 协议={e["protocol"]:.2f} / 训练靶={e["supervision"]:.2f} / 比值={e["ratio"]:.2f}x')

out = {'LIDC': LIDC_5, 'QUBIQ-brain': BRA_E, 'QUBIQ-kidney': KID_E,
       'agreement_index': _IDX}
os.makedirs(ANALYSIS_DIR, exist_ok=True)
with open(os.path.join(ANALYSIS_DIR, 'generalisation_table.json'), 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2)
print('\n已写: outputs/analysis/generalisation_table.json')
