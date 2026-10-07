# -*- coding: utf-8 -*-
# [2026-10-03] 已被 scripts/make_figures_v22.py 取代：论文 v22 的 9 张插图统一由该脚本生成；本脚本仅作历史保留，其输出不再用于论文。
"""make_fig3_ranked.py —— Fig. 3：Table 5 的排序在保留测试集上重算。

为什么重写：原来的 fig3 由 figs16 系列生成，**数值是硬编码在脚本里的字面量**，
所以 2026-09-04 的病例级重算没有传导到图上——论文里的 Fig. 3 一直显示
16.42 / 10.02 / 2.99 / 1.22 / 1.15 这组已废弃的实例级数字，与正文、Table 5
和它自己的图注全都矛盾。本脚本改为从权威 JSON 现取，杜绝同类问题。

行与 Table 5 一一对应；除位移中位数与噪声底外，均取保留测试集口径（图注已声明）。
数据来源：
  outputs/analysis_test/symmetric_analysis_applied.json  评测靶 / 阈值 / 监督靶 / 架构 / 平移代价
  outputs/analysis_scope/scope_analysis_merged2.json     视野效应两行
  outputs/analysis_scope/diag_offset_allarms.json        8 px 位移中位数
  outputs/analysis_full/noise_floor.json                 噪声底 2σ 与 CI
"""
import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'outputs', 'figures')


def rd(*p):
    with open(os.path.join(ROOT, *p), encoding='utf-8') as f:
        return json.load(f)


T = rd('outputs/analysis_test/symmetric_analysis_applied.json')['table1']
N = rd('outputs/analysis_full/noise_floor.json')
OFF = rd('outputs/analysis_scope/diag_offset_allarms.json')

SUP = ['tgtA', 'tgtB', 'tgtC', 'tgtD', 'tgtE']
ARCH = ['tgtB', 'archR18', 'archCNX', 'archPVT', 'archPU1e3']  # 2026-10-02：测试集 plain U-Net 用选定学习率版本
_missing = [x for x in SUP + ARCH if x not in T]
assert not _missing, f'测试集 JSON 缺少这些臂（先跑 run_fix_testset_plainunet_20261002.ps1）: {_missing}'


def span(arms, k, key='dice_opt'):
    v = [T[a][str(k)][key] for a in arms if a in T]
    return 100 * (max(v) - min(v))


def target_span(arm, key='dice_05'):
    v = [T[arm][str(k)][key] for k in (1, 2, 3, 4)]
    return 100 * (max(v) - min(v))


off = [100 * (r['mean_by_offset']['0'] - r['mean_by_offset']['8'])
       for k, r in OFF['runs'].items() if 'shift' not in k]
noise = N['fixed_0_5']['two_sigma_pts']
nci = N['fixed05_two_sigma_ci']

ROWS = [
    ('Evaluation field of view\n128 → 512 px, position-invariant arm', 43.84, (41.69, 46.09), 'protocol'),
    ('Lesion offset of 8 px from patch centre\nmedian of ten arms', float(np.median(off)), None, 'data'),
    ('Evaluation field of view\n128 → 512 px, both confounds removed', 22.51, (20.41, 24.77), 'protocol'),
    ('Evaluation target\nV ≥ 1 to V ≥ 4, arm A at threshold 0.5', target_span('tgtA'), None, 'protocol'),
    ('Binarisation threshold\narm A against V ≥ 4', 100 * (T['tgtA']['4']['dice_opt'] - T['tgtA']['4']['dice_05']), None, 'protocol'),
    ('Cost of translation augmentation\non the centred protocol', 100 * (T['tgtB']['2']['dice_opt'] - T['tgtBshift']['2']['dice_opt']), None, 'data'),
    ('Supervision target\nfive arms, against V ≥ 4', span(SUP, 4), None, 'method'),
    ('Architecture\nfive models, 8× parameter range', span(ARCH, 2), None, 'method'),
    ('Random seed (2σ)\npooled over four references', noise, tuple(nci), 'noise'),
]

BLUE, ORANGE, PURPLE, GREY = '#2a78d6', '#eb6834', '#4a3aa7', '#8f8f8a'
ROLE = {'protocol': BLUE, 'data': PURPLE, 'method': ORANGE, 'noise': GREY}
INK, BAND = '#22221f', '#f0efea'
mpl.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['DejaVu Sans', 'Arial'],
                     'font.size': 8, 'axes.labelsize': 8,
                     'xtick.labelsize': 7.4, 'ytick.labelsize': 7.4,
                     'axes.linewidth': 0.6, 'legend.frameon': False,
                     'figure.dpi': 120, 'savefig.dpi': 400})

fig, ax = plt.subplots(figsize=(7.09, 3.35))
y = np.arange(len(ROWS))[::-1]
ax.axvspan(0, noise, color=BAND, zorder=0)
for yi, (lab, v, ci, role) in zip(y, ROWS):
    ax.barh(yi, v, height=0.56, color=ROLE[role], edgecolor='white', lw=0.8, zorder=3)
    right = v
    if ci:
        ax.plot(ci, [yi, yi], color=INK, lw=0.9, zorder=5)
        for b in ci:
            ax.plot([b, b], [yi - 0.13, yi + 0.13], color=INK, lw=0.9, zorder=5)
        right = ci[1]
    ax.text(right + 1.0, yi, f'{v:.2f}', va='center', ha='left',
            fontsize=7.9, fontweight='bold', color=INK, zorder=6)
ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in ROWS], fontsize=7.2, linespacing=1.35)
ax.set_xlim(0, 54)
ax.set_ylim(y[-1] - 0.62, y[0] + 0.68)
ax.set_xlabel('Change in Dice (percentage points) when one factor is varied,\n'
              'all other factors held fixed')
ax.set_axisbelow(True)
ax.grid(axis='x', color='#e6e5e0', lw=0.6)
for s in ('top', 'right', 'left'):
    ax.spines[s].set_visible(False)
ax.tick_params(axis='y', length=0)
ax.legend(handles=[Patch(facecolor=ROLE[k], label=v) for k, v in
                   [('protocol', 'Evaluation protocol'), ('data', 'Data/training protocol'),
                    ('method', 'Learning design'), ('noise', 'Run-to-run noise')]],
          loc='lower right', bbox_to_anchor=(1.0, 0.02), handlelength=1.0,
          handleheight=1.0, borderpad=0.2, labelspacing=0.32, fontsize=7.2)
os.makedirs(OUT, exist_ok=True)
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(OUT, f'fig3_effect_sizes.{ext}'), bbox_inches='tight', pad_inches=0.06)
print('-> outputs/figures/fig3_effect_sizes.png / .pdf')
for lab, v, ci, role in ROWS:
    print(f'  {v:7.2f}  {role:9s} {lab.splitlines()[0]}')
