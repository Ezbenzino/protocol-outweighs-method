# -*- coding: utf-8 -*-
"""tost_sensitivity.py —— 等价检验对等价界的敏感性。

论文 §4.3 用 ±1.19 点（自己测出的噪声底）作为 TOST 的等价界。
"界本身是从同一批数据估出来的"是一个合理的审稿追问，
所以这里把结论在 ±1.0 / ±1.19 / ±1.5 三个界下各报一遍。

输出: outputs/analysis_full/tost_sensitivity.json
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FULL = os.path.join(ROOT, 'outputs', 'analysis_full')
TEST = os.path.join(ROOT, 'outputs', 'analysis_test')


def load(d):
    fs = [f for f in sorted(glob.glob(os.path.join(d, 'per_sample_*.csv')))
          if 'free' not in os.path.basename(f)]
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    df['arm'] = df.run.str.split('_fold').str[0]
    return df


def case_vals(df, arm, k, thr):
    m = (df.arm == arm) & (df[f'gt_area_v{k}'] > 0) & (np.isclose(df.threshold, thr, atol=1e-9))
    return df[m].groupby(['fold', 'case_id'])[f'dice_v{k}'].mean()


def tost(d, delta):
    """d: 逐病例配对差（Dice 点）。返回 (p_TOST, 是否在 alpha=0.05 下判等价)。"""
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    t1 = (d.mean() + delta) / se          # H0: mu <= -delta
    t2 = (d.mean() - delta) / se          # H0: mu >= +delta
    p1 = stats.t.sf(t1, n - 1)
    p2 = stats.t.cdf(t2, n - 1)
    p = float(max(p1, p2))
    return p, bool(p < 0.05)


def main():
    thr = json.load(open(os.path.join(FULL, 'symmetric_analysis.json'), encoding='utf-8'))['table1']
    out = {'bounds': [1.0, 1.19, 1.5], 'note': '±1.19 为论文采用的噪声底（fixed-0.5 pooled 2σ）',
           'contrasts': {}}
    for tag, d, thr_src in (('validation', load(FULL), thr), ('test', load(TEST), thr)):
        for k in (1, 2):
            a = case_vals(d, 'tgtC', k, thr_src['tgtC'][str(k)]['thr'])
            b = case_vals(d, 'tgtB', k, thr_src['tgtB'][str(k)]['thr'])
            a, b = a.align(b, join='inner')
            diff = (a - b).values * 100
            rec = {'n': int(len(diff)), 'mean_pts': round(float(diff.mean()), 3)}
            for delta in out['bounds']:
                p, eq = tost(diff, delta)
                rec[f'delta_{delta}'] = {'p_TOST': float(f'{p:.3g}'), 'equivalent': eq}
            out['contrasts'][f'C-B @G{k} ({tag})'] = rec
            flags = '  '.join(f"±{dl}: {'等价' if rec[f'delta_{dl}']['equivalent'] else '不判等价'}"
                              f" (p={rec[f'delta_{dl}']['p_TOST']:.2g})" for dl in out['bounds'])
            print(f"C-B @G{k} ({tag:10s}) n={rec['n']:4d} 差={rec['mean_pts']:+.2f} 点 | {flags}")
    with open(os.path.join(FULL, 'tost_sensitivity.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print('\n已写:', os.path.join(FULL, 'tost_sensitivity.json'))


if __name__ == '__main__':
    main()
