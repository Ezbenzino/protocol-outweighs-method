# -*- coding: utf-8 -*-
"""revise_paper_v14.py — v13 -> v14 修订
依据: docs/项目终审_20260903.md 第 1/2/3/4 项（1-4 处正文+格式修正）。

修改清单:
  1. [诚信级] 引言 "Survey method." 段: 删除不实陈述(双人抽取/标准数据库/回溯引文),
     改为指向 §3.10 的诚实概述。
  2. [硬格式] 摘要四段重写: 869 -> 248 词 (CIBM 上限 250, 不含 Keywords)。
  3. [硬格式] Highlight 第 1 条: 100 -> 84 字符 (CIBM 上限 85, 含 bullet)。
  4. [描述不准] Limitations: "trained on two folds" -> "evaluated on two folds"。
  5. [描述不准] Data availability: 按磁盘实际覆盖面重写 (11 主配置五折+测试集、
     两背景臂仅 fold0-1+测试集)。

用法: skin_seg python scripts/revise_paper_v14.py
输出: outputs/paper/论文_协议效应_v14_CIBM.docx
"""
import copy
import os
import sys

import docx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'outputs', 'paper', '论文_协议效应_v13_CIBM.docx')
DST = os.path.join(ROOT, 'outputs', 'paper', '论文_协议效应_v14_CIBM.docx')


def set_para_text(p, text):
    """清空段落所有 run，用第一个 run 的格式重建单一 run（保留 pPr/rPr）。"""
    rpr = None
    if p.runs:
        rpr = p.runs[0]._element.rPr
    for r in list(p.runs):
        r._element.getparent().remove(r._element)
    run = p.add_run(text)
    if rpr is not None:
        run._element.insert(0, copy.deepcopy(rpr))


# ---- 新文本 ----
NEW = {
    # [5]
    'abstract_bg': ('Background and objective. Reported Dice scores for lung nodule '
                    'segmentation on LIDC-IDRI span 0.43 to 0.99, far more than the '
                    '0.01\u20130.03 gains individual methods claim. Every study must choose a '
                    'label-fusion rule, a threshold, a patch-cropping rule and an evaluation '
                    'extent; most do not state them. We ask how much of this spread comes from '
                    'protocol rather than model capability, and whether probabilistic '
                    'supervision confers a benefit.'),
    # [6]
    'abstract_m': ('Methods. On LIDC-IDRI, four protocol factors \u2014 consensus level, '
                   'binarisation threshold, evaluation extent and nodule-centred patch '
                   'cropping \u2014 were controlled on 808 cases (five-fold) and 158 held-out '
                   'cases. Five supervision targets and five architectures were trained under '
                   'an otherwise identical 2-D pipeline, and effects were compared with a '
                   'repeated-training noise floor.'),
    # [7]
    'abstract_r': ('Results. Widening the evaluation window moved Dice by 22.51 points after '
                   'both confounds were removed; an 8-pixel window offset cost a median of '
                   '26.36 points. Changing only the evaluation target moved Dice by 16.42 '
                   'points and only the threshold by 10.02 points. On the held-out test set, '
                   'the largest learning-design effect was 3.24 points and the architecture '
                   'range 2.59 points, below the 1.15-point noise floor; protocol choices '
                   'exceeded learning design by 7.0\u00d7 and architecture by 8.7\u00d7. Soft '
                   'supervision over rater votes conferred no measurable benefit, and the '
                   'ordering replicated externally, falling with inter-rater agreement.'),
    # [8]
    'abstract_c': ('Conclusions. On LIDC-IDRI, evaluation-protocol choices dominate '
                   'data-preparation and learning-design choices by close to an order of '
                   'magnitude, and soft supervision over rater votes confers no measurable '
                   'benefit. Cross-study comparisons that omit these choices are not '
                   'interpretable.'),
    # [11]
    'highlight1': ('\u2022 Evaluation protocol moves LIDC nodule Dice far more than model '
                   'or learning design.'),
    # [20]
    'survey': ('Survey method. The survey covered peer-reviewed papers published between '
               '2018 and 2026 that report a segmentation result on LIDC-IDRI. Its search '
               'strategy, its access limitations, the three-state field coding and the fact '
               'that extraction was performed by a single rater are described in Section '
               '3.10; the per-paper extraction record is released with the code.'),
    # [271]
    'limitations': ('Evaluation scope rests on two folds. The field-of-view arms \u2014 '
                    'translation-augmented, background-sampled and merged \u2014 were '
                    'evaluated on two folds rather than five, for compute reasons. The paired '
                    'confidence intervals are computed over 158 test cases and are narrow, but '
                    'the fold-to-fold component of variance is estimated from two points.'),
    # [291]
    'data_avail': ('[PENDING: \u4ee3\u7801\u4ed3\u5e93 URL] The release contains the training '
                   'and evaluation code and the per-case Dice records. For the eleven main '
                   'configurations \u2014 six supervision-target arms, the position-robust arm '
                   'and four architecture arms \u2014 per-case records cover all five '
                   'validation folds and the held-out test set, each evaluated against all '
                   'four references over the full threshold sweep. The two background-sampling '
                   'arms cover validation folds 0\u20131 and the held-out test set, with their '
                   'field-of-view and window-offset ladder records. The noise-floor runs, the '
                   'effect-size and multiple-comparison analysis scripts, and the '
                   'literature-survey record with its per-paper field extraction are also '
                   'released. LIDC-IDRI is available from The Cancer Imaging Archive; the '
                   'QUBIQ 2021 data are available from grand-challenge.org under that '
                   "challenge's terms. No pixel data are redistributed."),
}


def main():
    if not os.path.exists(SRC):
        sys.exit(f'源 docx 不存在: {SRC}')
    # 备份目标（若存在旧 v14 则不覆盖，先归档）
    if os.path.exists(DST):
        bak = DST.replace('.docx', '_prev.docx')
        os.replace(DST, bak) if os.path.exists(bak) else os.rename(DST, bak)
    doc = docx.Document(SRC)
    paras = doc.paragraphs
    assert len(paras) == 348, f'段落数异常: {len(paras)} (期望 348)'

    # --- 1. 摘要四段 ---
    set_para_text(paras[5], NEW['abstract_bg'])
    set_para_text(paras[6], NEW['abstract_m'])
    set_para_text(paras[7], NEW['abstract_r'])
    set_para_text(paras[8], NEW['abstract_c'])
    # --- 2. Highlight 第 1 条 ---
    set_para_text(paras[11], NEW['highlight1'])
    # --- 3. Survey method 段 ---
    set_para_text(paras[20], NEW['survey'])
    # --- 4. Limitations ---
    set_para_text(paras[271], NEW['limitations'])
    # --- 5. Data availability ---
    set_para_text(paras[291], NEW['data_avail'])

    doc.save(DST)

    # --- 自检: 回读验证 ---
    doc2 = docx.Document(DST)
    p2 = doc2.paragraphs
    assert p2[5].text == NEW['abstract_bg'], '摘要 Background 回读不一致'
    assert p2[6].text == NEW['abstract_m'], '摘要 Methods 回读不一致'
    assert p2[7].text == NEW['abstract_r'], '摘要 Results 回读不一致'
    assert p2[8].text == NEW['abstract_c'], '摘要 Conclusions 回读不一致'
    assert p2[11].text == NEW['highlight1'], 'Highlight1 回读不一致'
    assert p2[20].text == NEW['survey'], 'Survey 回读不一致'
    assert p2[271].text == NEW['limitations'], 'Limitations 回读不一致'
    assert p2[291].text == NEW['data_avail'], 'Data availability 回读不一致'

    # 摘要词数自检 (不含 Keywords)
    n = sum(len(p2[i].text.split()) for i in (5, 6, 7, 8))
    print(f'摘要词数: {n} (上限 250)')
    assert n <= 250, '摘要超 250 词!'
    h1 = p2[11].text
    print(f'Highlight1 字符数: {len(h1)} (上限 85)')
    assert len(h1) <= 85, 'Highlight1 超 85 字符!'
    # 不实陈述残留自检
    for bad in ('verified by a second', 'backward citation chasing',
                'standard medical and engineering bibliographic databases'):
        assert bad not in p2[20].text, f'Survey 段仍含不实陈述: {bad}'
    print('v14 已写出并回读自检通过:', DST)


if __name__ == '__main__':
    main()
