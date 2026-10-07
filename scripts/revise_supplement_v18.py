# -*- coding: utf-8 -*-
"""revise_supplement_v18.py —— 补充材料从 v15 口径同步到 v18（病例级）。

补充材料是 2026-09-04 从 v15 拆出来的，之后正文经历了病例级重算（v16）
与两轮修订（v17/v18），补充材料一直没跟着改。结果是**正文与补充材料互相矛盾**：
  正文说噪声底 1.19、架构极差 0.96；补充材料说 1.15、1.22。
审稿人同时读两份就会发现。

本脚本把补充材料里所有随口径变化的数值同步到
outputs/analysis_full/symmetric_analysis.json 的病例级结果。
输出 outputs/paper/论文_协议效应_v18_补充材料.docx（文件名与正文版本对齐）。
"""
import json
import os

import docx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(ROOT, 'outputs', 'paper')
SRC = os.path.join(P, '论文_协议效应_v15_补充材料.docx')
DST = os.path.join(P, '论文_协议效应_v18_补充材料.docx')

V = json.load(open(os.path.join(ROOT, 'outputs/analysis_full/symmetric_analysis.json'),
                   encoding='utf-8'))['table1']

# 病例级 @V>=2 校准值与阈值
ARMS = {'A': 'tgtA', 'B': 'tgtB', 'C': 'tgtC', 'D': 'tgtD', 'E': 'tgtE'}
ARCH = ['archPU1e3', 'archR18', 'tgtB', 'archCNX', 'archPVT']
PRE = ['archR18', 'tgtB', 'archCNX', 'archPVT']


def d2(a):
    return V[a]['2']['dice_opt']


def thr(a):
    t = V[a]['2']['thr']
    return f'{t:.2f}'


TEXT = [
    # Note S1：噪声底
    ('Both are below the 1.15-point run-to-run noise floor established in Section 4.8',
     'Both are below the 1.19-point run-to-run noise floor established in Section 4.8'),
    # 说明"本工作的数值是病例级"
    ("Values for this work are five-fold cross-validation at each arm's calibrated threshold "
     "against G₂.",
     "Values for this work are five-fold cross-validation at each arm's calibrated threshold "
     "against G₂, with the case as the unit of analysis (Section 3.9)."),
    # 第一条观察
    ('Five supervision targets and five architectures together span 0.8570 to 0.8692, that is '
     '1.22 points; the four pretrained encoders span 0.27.',
     f'Five supervision targets and five architectures together span '
     f'{min(d2(a) for a in list(ARMS.values()) + ARCH):.4f} to '
     f'{max(d2(a) for a in list(ARMS.values()) + ARCH):.4f}, that is '
     f'{100 * (max(d2(a) for a in list(ARMS.values()) + ARCH) - min(d2(a) for a in list(ARMS.values()) + ARCH)):.2f} '
     f'points; the four pretrained encoders span '
     f'{100 * (max(d2(a) for a in PRE) - min(d2(a) for a in PRE)):.2f}.'),
]

# Table S1 的"本工作"区块
ROWS = {
    'Arm A, union training':        ('A', d2('tgtA'), thr('tgtA')),
    'Arm B, majority training':     ('B', d2('tgtB'), thr('tgtB')),
    'Arm C, majority + soft vote':  ('C', d2('tgtC'), thr('tgtC')),
    'Arm D, soft vote only':        ('D', d2('tgtD'), thr('tgtD')),
    'Arm E, SVLS':                  ('E', d2('tgtE'), thr('tgtE')),
}
RANGE_ROWS = {
    'Five architectures, 8× parameter range':
        f'{min(d2(a) for a in ARCH):.4f} – {max(d2(a) for a in ARCH):.4f}',
    '— of which the four pretrained encoders':
        f'{min(d2(a) for a in PRE):.4f} – {max(d2(a) for a in PRE):.4f}',
}


def run_replace(par, old, new):
    for r in par.runs:
        if old in r.text:
            r.text = r.text.replace(old, new)
            return True
    joined = ''.join(r.text for r in par.runs)
    if old not in joined:
        return False
    par.runs[0].text = joined.replace(old, new)
    for r in par.runs[1:]:
        r.text = ''
    return True


def setcell(cell, text):
    p = cell.paragraphs[0]
    if p.runs:
        for r in list(p.runs)[1:]:
            r._element.getparent().remove(r._element)
        p.runs[0].text = text
    else:
        p.add_run(text)


def main():
    doc = docx.Document(SRC)
    for old, new in TEXT:
        assert any(run_replace(p, old, new) for p in doc.paragraphs), f'未找到: {old[:60]!r}'

    tb = doc.tables[0]
    hit = 0
    for row in tb.rows:
        lab = row.cells[0].text.strip()
        if lab in ROWS:
            _, dsc, t = ROWS[lab]
            setcell(row.cells[1], f'{dsc:.4f}')
            setcell(row.cells[5], f'calibrated, {t}')
            hit += 1
        elif lab in RANGE_ROWS:
            setcell(row.cells[1], RANGE_ROWS[lab])
            hit += 1
    assert hit == 7, f'Table S1 只匹配到 {hit} 行，应为 7'

    doc.save(DST)

    d2doc = docx.Document(DST)
    txt = '\n'.join(p.text for p in d2doc.paragraphs)
    tbl = '\n'.join(' | '.join(c.text for c in r.cells) for r in d2doc.tables[0].rows)
    for bad in ('1.15-point', '0.8582', '0.8686', '0.8570 to 0.8692', '1.22 points',
                'span 0.27', '0.8664 – 0.8692'):
        assert bad not in txt + tbl, f'补充材料仍含旧值: {bad}'
    for good in ('1.19-point', '0.8670', '0.8740', '0.8752', '0.8746', '0.8761',
                 'the case as the unit of analysis'):
        assert good in txt + tbl, f'补充材料缺少: {good}'
    print('Table S1（本工作区块）已同步为病例级：')
    for r in d2doc.tables[0].rows[2:9]:
        print('   ' + ' | '.join(c.text for c in r.cells)[:120])
    print('\n回读自检通过 ->', DST)


if __name__ == '__main__':
    main()
