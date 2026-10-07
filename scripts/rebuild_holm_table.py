# -*- coding: utf-8 -*-
"""从病例级主分析 symmetric_analysis.json（table2，配对 t 检验）重建 Holm 校正表。

背景（2026-10-02 审计）：outputs/analysis_full/holm_table3.json 生成于 2026-09-02，早于 09-04 的
"分析单元统一到病例级"，其中 12/20 个对比的差值、CI、p、dz 与论文 Table 4（病例级）不一致，
却作为"Table 4 的 Holm 校正"被放进了 release。本脚本按论文 §3.9 的定义重建：
20 个预设对比 = {C−A, B−A, C−B, D−C, D−B} × {V≥1..V≥4}，Holm 逐级下调校正（m = 20）。
输出字段与旧文件保持一致，并补 diff_pts / ci_pts 两个以"Dice 点"为单位的字段。
用法：python scripts/rebuild_holm_table.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'outputs' / 'analysis_full' / 'symmetric_analysis.json'
DST = ROOT / 'outputs' / 'analysis_full' / 'holm_table3.json'

t2 = json.load(open(SRC, encoding='utf-8'))['table2']
PAIRS = ['tgtC-tgtA', 'tgtB-tgtA', 'tgtC-tgtB', 'tgtD-tgtC', 'tgtD-tgtB']
rows = []
for pair in PAIRS:
    for ref in ('1', '2', '3', '4'):
        r = t2[pair][ref]
        rows.append({'pair': pair, 'ref': int(ref), 'n': r['n'],
                     'diff': r['diff'] * 100, 'p': r['p_t'], 'ci': r['ci'], 'dz': r['dz'],
                     'diff_pts': round(r['diff'] * 100, 4),
                     'ci_pts': [round(r['ci'][0] * 100, 4), round(r['ci'][1] * 100, 4)]})
m = len(rows)
assert m == 20
order = sorted(range(m), key=lambda i: rows[i]['p'])
running = 0.0
for rank, i in enumerate(order):
    running = max(running, min(1.0, (m - rank) * rows[i]['p']))
    rows[i]['holm'] = running
for r in rows:
    r['sig_raw'] = r['p'] < 0.05
    r['sig_holm'] = r['holm'] < 0.05
n_raw = sum(r['sig_raw'] for r in rows)
n_holm = sum(r['sig_holm'] for r in rows)
json.dump(rows, open(DST, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print(f'wrote {DST}  (20 contrasts; significant raw {n_raw}, after Holm {n_holm})')
for r in rows:
    print(f"  {r['pair']:10s} V>={r['ref']}  {r['diff_pts']:+.2f} [{r['ci_pts'][0]:+.2f},{r['ci_pts'][1]:+.2f}]"
          f"  p={r['p']:.1e}  dz={r['dz']:+.3f}  holm={r['holm']:.1e}")
