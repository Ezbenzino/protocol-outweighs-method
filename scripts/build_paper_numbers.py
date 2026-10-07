# -*- coding: utf-8 -*-
"""build_paper_numbers.py —— 把各分析产物汇总成论文插图用的单一 JSON。

此前 outputs/analysis/paper_numbers_v11.json 没有生成脚本（无法复现）。
本脚本从权威产物重建它，键名与旧文件一致，供 figs11 系列插图脚本直接读取。

数据来源（全部为病例级口径，见 docs/分析单元审计_20260904.md）：
  outputs/analysis_full/symmetric_analysis.json          -> /val
  outputs/analysis_test/symmetric_analysis_applied.json  -> /test
  outputs/analysis_scope/scope_analysis_merged2.json     -> /scope
  outputs/analysis_full/noise_floor.json                 -> /noise
  outputs/analysis/disagreement_index.json               -> /disagree
  outputs/analysis/generalisation_table.json             -> /gen
  outputs/analysis/dataset_stats.json                    -> /stats
  outputs/analysis_scope/diag_offset_allarms.json        -> /offset (+ /offset_summary)
  outputs/analysis_full/per_sample_val_fold*.csv         -> /curves（病例级阈值-Dice 曲线）

用法: python scripts/build_paper_numbers.py
"""
import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'outputs', 'analysis', 'paper_numbers_v11.json')


def rd(*parts):
    with open(os.path.join(ROOT, *parts), encoding='utf-8') as f:
        return json.load(f)


def curves_case_level():
    """病例级阈值-Dice 曲线：病例内对实例平均 -> 折内平均 -> 跨折平均。

    每个臂给出与阈值网格对齐的 threshold / dice_v1..4 / area_pred，
    以及该臂样本上的病例级平均 GT 面积 gt_area_v1..4。
    """
    fs = [f for f in sorted(glob.glob(os.path.join(
        ROOT, 'outputs', 'analysis_full', 'per_sample_val_fold*.csv')))
        if 'free' not in os.path.basename(f)]
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    df['arm'] = df.run.str.split('_fold').str[0]
    thrs = sorted(df.threshold.unique())
    out = {}
    for arm, g in df.groupby('arm'):
        rec = {'threshold': [float(t) for t in thrs]}
        for k in (1, 2, 3, 4):
            s = g[g[f'gt_area_v{k}'] > 0]
            c = (s.groupby(['fold', 'threshold', 'case_id'])[f'dice_v{k}'].mean()
                 .groupby(['fold', 'threshold']).mean().groupby('threshold').mean()
                 .reindex(thrs))
            rec[f'dice_v{k}'] = [float(x) for x in c.values]
            a = (s.groupby(['fold', 'case_id'])[f'gt_area_v{k}'].mean()
                 .groupby('fold').mean().mean())
            rec[f'gt_area_v{k}'] = [float(a)] * len(thrs)
        ap = (g.groupby(['fold', 'threshold', 'case_id']).area_pred.mean()
              .groupby(['fold', 'threshold']).mean().groupby('threshold').mean()
              .reindex(thrs))
        rec['area_pred'] = [float(x) for x in ap.values]
        out[arm] = rec
    return out


def offset_summary(off):
    drops = {}
    for run, r in off['runs'].items():
        m = r['mean_by_offset']
        drops[run] = (m['0'] - m['8']) * 100
    noaug = {k: v for k, v in drops.items() if 'shift' not in k}
    vals = np.array(sorted(noaug.values()))
    return {'drops': drops, 'n_noaug': int(len(vals)),
            'median_noaug': float(np.median(vals)),
            'min': float(vals.min()), 'max': float(vals.max())}


def main():
    off = rd('outputs', 'analysis_scope', 'diag_offset_allarms.json')
    rep = {
        'val': rd('outputs', 'analysis_full', 'symmetric_analysis.json'),
        'scope': rd('outputs', 'analysis_scope', 'scope_analysis_merged2.json'),
        'noise': rd('outputs', 'analysis_full', 'noise_floor.json'),
        'disagree': rd('outputs', 'analysis', 'disagreement_index.json'),
        'gen': rd('outputs', 'analysis', 'generalisation_table.json'),
        'test': rd('outputs', 'analysis_test', 'symmetric_analysis_applied.json'),
        'stats': rd('outputs', 'analysis', 'dataset_stats.json'),
        'offset': off,
        'offset_summary': offset_summary(off),
        'curves': curves_case_level(),
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print('已写:', OUT)
    print('  val 臂数   :', len(rep['val']['table1']))
    print('  test 臂数  :', len(rep['test']['table1']))
    print('  curves 臂数:', len(rep['curves']))
    print('  位移中位数 :', round(rep['offset_summary']['median_noaug'], 2),
          f"（{rep['offset_summary']['n_noaug']} 个未增强臂）")
    print('  一致性指数 :', rep['disagree']['LIDC']['area_mean_ratio'])
    print('  噪声 2sigma:', rep['noise']['fixed_0_5']['two_sigma_pts'])


if __name__ == '__main__':
    main()
