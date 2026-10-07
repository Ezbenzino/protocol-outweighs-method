# -*- coding: utf-8 -*-
# [2026-10-03] 已被 scripts/make_figures_v22.py 取代：论文 v22 的 9 张插图统一由该脚本生成；本脚本仅作历史保留，其输出不再用于论文。
"""fig8_evaluation_scope：视野阶梯折线图（4 臂，含合并臂）。
数据源：outputs/analysis_scope/scope_analysis_merged2.json -> table1_ladder
口径与论文 §4.11/§4.13 一致：验证集校准阈值、病例级均值、G2 (V>=2)。
论文定格：tgtBshift win128->win512 = -43.84 pts；merged = -22.51 pts。
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = r'outputs/figures'
aj = json.load(open(r'outputs/analysis_scope/scope_analysis_merged2.json'))
t1 = aj['table1_ladder']

SCOPES = ['patch128', 'win128', 'win192', 'win256', 'win384', 'win512', 'slice', 'volume']
XLAB = ['patch\n128', 'win\n128', 'win\n192', 'win\n256', 'win\n384', 'win\n512', 'slice', 'volume']

ARMS = {
    'tgtB':         ('Centre-biased (arm B)',         '#2a78d6'),
    'bg_maj':       ('Background-sampled (arm BG)',   '#1baf7a'),
    'tgtBshift':    ('Translation-augmented (arm Sh)', '#eb6834'),
    'tgtBshift_bg': ('Shift + background (merged)',   '#4a3aa7'),
}

fig, ax = plt.subplots(figsize=(8.6, 3.1))
x = np.arange(len(SCOPES))
for arm, (label, color) in ARMS.items():
    row = t1[f'{arm}_v2']
    vals = np.array([row[s] for s in SCOPES])
    ax.plot(x, vals, marker='o', ms=4.2, lw=1.6, color=color, label=label)
    ax.plot([1], [vals[1]], 'o', ms=6.5, color=color, mec='white', mew=1.0, zorder=5)

# 标注 win128 -> win512 的 step（论文口径）
d_sh = (t1['tgtBshift_v2']['win128'] - t1['tgtBshift_v2']['win512']) * 100
d_mg = (t1['tgtBshift_bg_v2']['win128'] - t1['tgtBshift_bg_v2']['win512']) * 100
ax.annotate(f'-{d_mg:.1f} pts (merged)',
            xy=(5, t1['tgtBshift_bg_v2']['win512']),
            xytext=(4.1, 0.30), fontsize=6.4, color='#4a3aa7',
            arrowprops=dict(arrowstyle='-', color='#4a3aa7', lw=0.6))
ax.annotate(f'-{d_sh:.1f} pts (shift)',
            xy=(5, t1['tgtBshift_v2']['win512']),
            xytext=(2.2, 0.16), fontsize=6.4, color='#eb6834',
            arrowprops=dict(arrowstyle='-', color='#eb6834', lw=0.6))
ax.axvline(0.5, color='#8a8983', lw=0.7, ls=(0, (3, 2)))
ax.text(0.5, 0.03, 'training\ncrop', ha='center', va='bottom', fontsize=6, color='#52514e')
ax.set_xticks(x, XLAB, fontsize=6.5)
ax.set_xlabel('Evaluation field of view', fontsize=7.5)
ax.set_ylabel('Dice (V $\\geq$ 2, calibrated thr.)', fontsize=7.5)
ax.set_ylim(0, 0.92)
ax.grid(True, axis='y', color='#e4e3de', lw=0.5, zorder=0)
ax.set_axisbelow(True)
for sp in ax.spines.values():
    sp.set_color('#8a8983')
ax.tick_params(labelsize=7)
# 图例移到轴外右侧，避免与左下 'training crop' 标注及曲线重叠
ax.legend(loc='center left', fontsize=6.4, frameon=False, bbox_to_anchor=(1.01, 0.5))
ax.set_title('Field-of-view cost persists after both training-side remedies '
             f'(win128 $\\rightarrow$ win512: $-$43.8 pts shift-only vs $-$22.5 pts merged)',
             loc='left', fontsize=8, pad=6)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(OUT, f'fig8_evaluation_scope.{ext}'), dpi=300)
print(f'fig8 done: shift={d_sh:.2f} merged={d_mg:.2f}')
