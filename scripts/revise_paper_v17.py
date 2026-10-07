# -*- coding: utf-8 -*-
"""revise_paper_v17.py —— v16 -> v17：两处漏改的旧值 + 四处审稿预防性补充。

两处漏改（v16 那一遍的关键词表没覆盖到，由"旧分析独有数值"全库扫描查出）：
  1. §4.7 面积匹配段仍是实例级数值
  2. §4.9 测试集"对角结构"要点仍是实例级数值

四处补充（依据 docs/补强方案评估_20260904.md 的审稿预设问答）：
  3. §4.1 新增一段，把"评测靶效应是不是纯面积算术"这一问就地答掉
  4. §3.9 新增分析单元敏感性说明（两种口径的数字并列）
  5. §4.3 补等价界敏感性（±1.0 / ±1.19 / ±1.5 结论一致，源 tost_sensitivity.json）
  6. §5.1 点名方法轴的四个子轴，答"方法侧采样不足"

数值来源：outputs/analysis_full/symmetric_analysis.json、
         outputs/analysis_test/symmetric_analysis_applied.json、
         outputs/analysis_full/tost_sensitivity.json
"""
import copy
import json
import os

import docx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(ROOT, 'outputs', 'paper')
SRC = os.path.join(P, '论文_协议效应_v16_CIBM.docx')
DST = os.path.join(P, '论文_协议效应_v17_CIBM.docx')

V = json.load(open(os.path.join(ROOT, 'outputs/analysis_full/symmetric_analysis.json'), encoding='utf-8'))
T = json.load(open(os.path.join(ROOT, 'outputs/analysis_test/symmetric_analysis_applied.json'), encoding='utf-8'))

REPLACE = [
    # ---- 1. §4.7 面积匹配（病例级重算）----
    ('Setting each arm to the threshold at which its mean predicted area is closest to the mean '
     'area of G₂ gives 206.3, 204.0, 203.9 and 203.5 px for arms A to D, at which points Dice '
     'against G₂ is 0.8534, 0.8653, 0.8661 and 0.8672 — arms B, C and D within 0.002 of each '
     'other, arm A 1.2 to 1.4 points behind.',
     'Setting each arm to the threshold at which its mean predicted area is closest to the '
     'case-level mean area of G₂ (163.4 px) gives 161.4, 163.8, 163.1 and 163.3 px for arms A to '
     'D, at which points Dice against G₂ is 0.8647, 0.8732, 0.8740 and 0.8737 — arms B, C and D '
     'within 0.0008 of each other, arm A about 0.9 points behind.'),
    # ---- 2. §4.9 测试集对角结构 ----
    ('Arm A remained best against G₁ (0.8662) and arm B against G₃ (0.8410) and G₄ (0.8093); '
     'arms B, C and D were within 0.0011 of one another against G₂.',
     'Arm A remained best against G₁ (0.8677) and arm B against G₃ (0.7738) and G₄ (0.6941); '
     'arms B, C and D were within 0.0014 of one another against G₂.'),
    # ---- 1b. §4.3 高共识处的危害值（v16 遗漏，实例级->病例级）----
    ('−0.51 points against G₃ ([−0.73, −0.29], p = 8.2 × 10⁻⁶)',
     '−0.55 points against G₃ ([−0.81, −0.29], p = 3.8 × 10⁻⁵)'),
    # ---- 1c. §4.3 D−B / D−C 三处（v16 遗漏）----
    ('+0.25, +0.01, −0.62 and −2.76 points against G₁ to G₄',
     '+0.25, +0.06, −0.64 and −2.65 points against G₁ to G₄'),
    ('shows no significant difference on any reference except a marginal −0.28 against G₄ (p = 0.056).',
     'shows no significant difference on any reference, including G₄ (−0.18 points, p = 0.39).'),
    # ---- 5. §4.3 等价界敏感性 ----
    ('the soft-supervision effect is therefore not merely undetected but statistically equivalent '
     'to the resolution of the measurement.',
     'the soft-supervision effect is therefore not merely undetected but statistically equivalent '
     'to the resolution of the measurement. The conclusion does not rest on the bound being the '
     'measured noise floor: both contrasts are declared equivalent at ±1.0 and at ±1.5 points as '
     'well, with p ≤ 2.8 × 10⁻¹¹ in every case, and the same holds on the held-out test set.'),
    # ---- 6. §5.1 方法轴的四个子轴 ----
    ('Against that, a controlled comparison values the largest architectural change measured here '
     'at 0.96 points across an 8-fold parameter range, and the largest difference among five '
     'defensible supervision strategies at 2.65.',
     'Against that, a controlled comparison values the largest architectural change measured here '
     'at 0.96 points across an 8-fold parameter range, and the largest difference among five '
     'defensible supervision strategies at 2.65. The learning-design axis was not sampled '
     'narrowly in reaching that figure: four of its dimensions were varied — the supervision '
     'target, the architecture, the loss composition (Supplementary Note S1) and the encoder '
     'learning rate (Table 6) — and the largest effect on any of them is the 2.65 points above, '
     'with the other three at or below the run-to-run noise floor.'),
]

# ---- 3. §4.1 新增段：区分"算术部分"与"非算术部分" ----
ARITH = (
    'How much of this is arithmetic. The four references differ in extent by construction — their '
    'case-level mean areas are 260.3, 202.4, 142.8 and 89.3 pixels, so the union reference covers '
    '2.9 times the area of the unanimous one — and a prediction calibrated to one extent must lose '
    'Dice against another. Part of the 25.58 points is therefore geometry rather than anything '
    'about the model, and the calibrated column separates the two: scoring each reference at its '
    'own optimal threshold is precisely the correction that rescales predicted extent to match the '
    'reference, and it leaves 16.39 of the 25.58 points standing. Roughly two thirds of the effect '
    'survives the operation that an area-mismatch account predicts would remove it. The '
    'area-matched operating point of Section 4.7 reaches the same conclusion from the other '
    'direction.')

# ---- 4. §3.9 新增段：分析单元敏感性 ----
UNIT = (
    'Sensitivity to the unit of analysis. The number of nodule instances per case is highly '
    'unequal — a median of 9 and a maximum of 79 — so the choice of unit is not cosmetic. Under '
    'the alternative convention, in which each instance is treated as an observation, the '
    'protocol effects are systematically smaller and the learning-design effects slightly larger: '
    'the evaluation-target effect of Section 4.1 becomes 16.42 rather than 25.58 points, the '
    'threshold effect 10.02 rather than 9.42, the supervision-target range 2.99 rather than 2.65 '
    'and the architecture range 1.22 rather than 0.96. The ordering, and every conclusion drawn '
    'from it, is unchanged under either convention. We report the case-level convention because '
    'instances within a patient are not independent; the released per-case records allow either '
    'to be recomputed.')


def run_replace(par, old, new, tag):
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


def insert_after(doc, anchor_idx, text):
    """在 anchor 段之后插入一个同样式的新段落。"""
    src = doc.paragraphs[anchor_idx]
    new_p = copy.deepcopy(src._element)
    src._element.addnext(new_p)
    par = docx.text.paragraph.Paragraph(new_p, src._parent)
    for r in list(par.runs)[1:]:
        r._element.getparent().remove(r._element)
    par.runs[0].text = text
    return par


def main():
    doc = docx.Document(SRC)
    done = 0
    for old, new in REPLACE:
        hit = False
        for par in doc.paragraphs:
            if run_replace(par, old, new, ''):
                hit = True
                done += 1
                break
        assert hit, f'未找到目标文本: {old[:70]!r}'

    ps = [p.text for p in doc.paragraphs]
    i_arith = [i for i, t in enumerate(ps) if t.strip().startswith('Sensitivity to the case set')][0]
    insert_after(doc, i_arith, ARITH)

    ps = [p.text for p in doc.paragraphs]
    i_unit = [i for i, t in enumerate(ps)
              if t.strip().startswith('The unit of analysis is the case, not the patch')][0]
    insert_after(doc, i_unit, UNIT)

    doc.save(DST)

    d2 = docx.Document(DST)
    txt = '\n'.join(p.text for p in d2.paragraphs)
    for bad in ('0.8653', '0.8672', '206.3', '204.0', '203.9', '203.5',
                '0.8662', '0.8410', '0.8093', 'within 0.0011',
                '8.2 × 10⁻⁶', '−2.76 points', 'p = 0.056'):
        assert bad not in txt, f'v17 仍含旧值: {bad}'
    for good in ('How much of this is arithmetic', 'Sensitivity to the unit of analysis',
                 '±1.0 and at ±1.5 points', 'four of its dimensions were varied',
                 '0.8647, 0.8732, 0.8740 and 0.8737', '0.8677', '0.7738', '0.6941',
                 '3.8 × 10⁻⁵', '−2.65 points against G₁ to G₄', 'including G₄ (−0.18 points'):
        assert good in txt, f'v17 缺少: {good}'
    a = [i for i, p in enumerate(d2.paragraphs) if p.text.strip() == 'Abstract'][0]
    h = [i for i, p in enumerate(d2.paragraphs) if p.text.strip() == 'Highlights'][0]
    n = sum(len(p.text.split()) for p in d2.paragraphs[a + 1:h]
            if p.text.strip() and not p.text.strip().startswith('Keywords'))
    print(f'替换 {done} 处，新增 2 段；摘要仍 {n} 词')
    print(f'段落 {len(d2.paragraphs)}  表 {len(d2.tables)}  图 {len(d2.inline_shapes)}  '
          f"PENDING {txt.count('PENDING')}")
    print(f'正文词数 {sum(len(p.text.split()) for p in d2.paragraphs)}')
    print(DST)


if __name__ == '__main__':
    main()
