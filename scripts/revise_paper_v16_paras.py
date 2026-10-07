# -*- coding: utf-8 -*-
"""revise_paper_v16_paras.py —— v15 -> v16 第 1 遍：正文段落数字改为病例级口径。

依据 docs/分析单元审计_20260904.md：论文原先的点估计（表1/4/5/6 系列）是
【结节实例级】均值，而配对检验、视野效应是【病例级】，头条比值把两种口径相除。
本遍把全部正文数字统一到病例级，数值取自重跑后的：
  outputs/analysis_full/symmetric_analysis.json
  outputs/analysis_test/symmetric_analysis_applied.json
  outputs/analysis_full/noise_floor.json
  outputs/analysis/{disagreement_index,generalisation_table}.json

表格与插图在第 2 遍（revise_paper_v16_tables.py）处理。
"""
import os
import docx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(ROOT, 'outputs', 'paper')
SRC = os.path.join(P, '论文_协议效应_v15_CIBM.docx')
DST = os.path.join(P, '论文_协议效应_v16_CIBM.docx')

# (段落号, 原文片段, 新文片段)
EDITS = [
 (5, 'moved Dice by 16.42 points and only the threshold by 10.02 points',
     'moved Dice by 25.58 points and only the threshold by 9.42 points'),
 (5, 'the largest learning-design effect was 3.24 points and the architecture range 2.59 points, below the 1.15-point noise floor; protocol choices exceeded learning design by 7.0× and architecture by 8.7×',
     'the largest learning-design effect was 3.12 points and the architecture range 1.87 points, against a 1.19-point noise floor; protocol choices exceeded learning design by 7.2× and architecture by 12.0×'),
 (29, 'by 7.0× and the architecture range by 8.7× on the conservative estimate',
      'by 7.2× and the architecture range by 12.0× on the conservative estimate'),
 (67, 'Across the full 792-case cohort, 64.7 per cent of contributing cases contain no pixel at all on which all four readers agree.',
      'Across the full 792-case cohort, 17.8 per cent of contributing cases contain no pixel anywhere on which all four readers agree.'),
 (98, 'because on LIDC 64.7 per cent of cases have an empty unanimous reference and a per-case ratio is dominated by that point mass',
      'because on LIDC the unanimous reference is empty for 17.8 per cent of cases and for 47.1 per cent of individual lesions, and a per-case ratio is dominated by that point mass'),
 (114, '4.1 The evaluation target alone moves Dice by up to 16.42 points',
       '4.1 The evaluation target alone moves Dice by up to 25.58 points'),
 (115, 'scores 0.8545 against G₁ and 0.6903 against G₄ — a spread of 16.42 Dice points. The three remaining arms span 11.17 to 11.66 points',
       'scores 0.8648 against G₁ and 0.6090 against G₄ — a spread of 25.58 Dice points. The three remaining arms span 20.58 to 20.97 points'),
 (116, 'the spreads become 6.77, 6.50, 9.30 and 9.60 points for arms A to D',
       'the spreads become 16.39, 15.92, 18.53 and 18.62 points for arms A to D'),
 (116, 'but the additional 2 to 10 points present at a fixed threshold',
       'but the additional 2 to 9 points present at a fixed threshold'),
 (117, 'The result is unchanged in substance: 16.45 points at a fixed threshold of 0.5 and 6.70 at each reference’s own optimum, versus 16.42 and 6.64 on the full set (Table 2).',
       'The result is unchanged in substance: 25.81 points at a fixed threshold of 0.5 and 16.96 at each reference’s own optimum, versus 25.58 and 16.39 on the full set (Table 2).'),
 (119, 'Against G₂ the majority and consensus arms peak at 0.50, the soft-only arm at 0.25, and the union arm at 0.99. Against G₄ the optima are 0.998, 0.98, 0.97 and 0.999.',
       'Against G₂ the majority arm peaks at 0.55, the consensus arm at 0.65, the soft-only arm at 0.50 and the union arm at 0.99. Against G₄ the optima are 0.999 for the union arm and 0.998, 0.98 and 0.98 for arms B, C and D.'),
 (120, 'moving from a threshold of 0.5 to its optimum is worth 10.02 Dice points. The full range across all four supervision targets, each at its own calibrated threshold, is at most 2.99 points (against G₄) and as little as 0.58 points (against G₃). Choosing the threshold matters roughly three times more',
       'moving from a threshold of 0.5 to its optimum is worth 9.42 Dice points. The full range across all four supervision targets, each at its own calibrated threshold, is at most 2.65 points (against G₄) and as little as 0.64 points (against G₃). Choosing the threshold matters roughly three and a half times more'),
 (129, 'and +0.11 against G₂ ([−0.12, +0.34], p = 0.35, dz = 0.037)',
       'and +0.12 against G₂ ([−0.11, +0.34], p = 0.32, dz = 0.040)'),
 (129, 'Section 4.8 provides an independent check: +0.11 points is about one tenth of the noise floor',
       'Section 4.8 provides an independent check: +0.12 points is about one tenth of the noise floor'),
 (129, 'Two one-sided tests with an equivalence bound of ±1.15 points',
       'Two one-sided tests with an equivalence bound of ±1.19 points'),
 (132, 'against G₂ the total is +0.81 points, of which the mask-extent component (B − A) accounts for +0.70 and the soft-supervision component (C − B) for +0.11',
       'against G₂ the total is +0.82 points, of which the mask-extent component (B − A) accounts for +0.70 and the soft-supervision component (C − B) for +0.12'),
 (138, 'Arm A is best against the union reference (0.8569 versus 0.8504–0.8531); arm B is best against V ≥ 3 (0.8258) and V ≥ 4 (0.8025); against V ≥ 2 arms B, C and D are within 0.0011 of one another (0.8675, 0.8684, 0.8686) while arm A trails by 0.9 to 1.0 points.',
       'Arm A is best against the union reference (0.8661 versus 0.8586–0.8611); arm B is best against V ≥ 3 (0.7740) and V ≥ 4 (0.7149); against V ≥ 2 arms B, C and D are within 0.0012 of one another (0.8740, 0.8752, 0.8746) while arm A trails by 0.7 to 0.8 points.'),
 (140, 'gives 0.8582 ± 0.0121 for A, 0.8675 ± 0.0096 for B, 0.8684 ± 0.0103 for C and 0.8686 ± 0.0097 for D',
       'gives 0.8670 ± 0.0090 for A, 0.8740 ± 0.0073 for B, 0.8752 ± 0.0079 for C and 0.8746 ± 0.0065 for D'),
 (146, 'by 7.0× and the architecture range by 8.7×; on the position-invariant arm the ratios are 13.5× and 16.9×',
       'by 7.2× and the architecture range by 12.0×; on the position-invariant arm the ratios are 14.1× and 23.4×'),
 (147, 'On validation the architecture range is 1.1 times the noise floor, and within the four ImageNet-pretrained encoders it falls below it; on the test set it widens to 2.59 points (Section 4.6).',
       'On validation the architecture range is 0.8 times the noise floor — below it — and within the four ImageNet-pretrained encoders far below; on the test set it widens to 1.87 points (Section 4.6).'),
 (151, 'the evaluation-target effect is 18.00 rather than 16.42 points, the supervision-target range 3.24 rather than 2.99 and the architecture range 2.59 rather than 1.22',
       'the evaluation-target effect is 28.58 rather than 25.58 points, the supervision-target range 3.12 rather than 2.65 and the architecture range 1.87 rather than 0.96'),
 (152, 'evaluation target 18.00 points, supervision target 3.24, architecture 2.59',
       'evaluation target 28.58 points, supervision target 3.12, architecture 1.87'),
 (158, 'The range across all five architectures is 1.22 Dice points at V ≥ 2, between 0.84 and 1.46 points across the four references',
       'The range across all five architectures is 0.96 Dice points at V ≥ 2, between 0.68 and 1.37 points across the four references'),
 (158, 'differ by 0.27 points at V ≥ 2 (0.8664 to 0.8692), which is below the 1.15-point run-to-run noise floor',
       'differ by 0.19 points at V ≥ 2 (0.8735 to 0.8754), which is below the 1.19-point run-to-run noise floor'),
 (159, 'left the architecture range at 1.05 and 1.22 points at V ≥ 2 respectively, both comparable to the noise floor',
       'left the architecture range at 0.86 and 0.96 points at V ≥ 2 respectively, both below the noise floor'),
 (169, 'the standard deviation was 0.92 and 0.53 Dice points for arms A and B on fold 0, and 0.33 and 0.27 points on fold 1, and 0.57 points pooled across both arms and both folds (2 arms × 2 folds × 4 targets × 3 seeds, 32 degrees of freedom; 95% CI [0.46, 0.76]). Taking 2σ as a practical resolution limit gives 1.15 Dice points (95% CI [0.92, 1.52])',
       'the standard deviation was 0.93 and 0.56 Dice points for arms A and B on fold 0, and 0.38 and 0.31 points on fold 1, and 0.60 points pooled across both arms and both folds (2 arms × 2 folds × 4 targets × 3 seeds, 32 degrees of freedom; 95% CI [0.48, 0.79]). Taking 2σ as a practical resolution limit gives 1.19 Dice points (95% CI [0.97, 1.59])'),
 (171, 'Its own confidence interval spans [0.92, 1.52]. Quoting it to two decimals is a convenience of notation, not a claim of precision, and a difference of 1.1 points should not be described as "above the floor" without that caveat.',
       'Its own confidence interval spans [0.97, 1.59]. Quoting it to two decimals is a convenience of notation, not a claim of precision, and a difference of 1.2 points should not be described as "above the floor" without that caveat.'),
 (172, '2σ is 1.29, 0.70, 1.18 and 1.31 points at V ≥ 1 to 4; when each model selects its own optimal threshold it is 0.77, 0.63, 0.56 and 0.83 points, pooling to 0.70. The pooled fixed-threshold value of 1.15',
       '2σ is 1.19, 0.85, 1.27 and 1.39 points at V ≥ 1 to 4; when each model selects its own optimal threshold it is 0.74, 0.49, 0.50 and 0.65 points, pooling to 0.60. The pooled fixed-threshold value of 1.19'),
 (175, 'and the point estimate of +0.11 is about a tenth of the run-to-run noise floor',
       'and the point estimate of +0.12 is about a tenth of the run-to-run noise floor'),
 (175, 'The architecture range sits at the floor, and we treat it accordingly',
       'The architecture range sits at or below the floor, and we treat it accordingly'),
 (180, 'was 18.00 points for arm A, and the largest learning-design effect — the supervision target — was 3.24 points. The architecture range at V ≥ 2 was 2.59 points.',
       'was 28.58 points for arm A, and the largest learning-design effect — the supervision target — was 3.12 points. The architecture range at V ≥ 2 was 1.87 points.'),
 (181, 'and +0.11 against G₂ ([−0.05, +0.27], p = 0.20)',
       'and +0.10 against G₂ ([−0.06, +0.26], p = 0.22)'),
 (182, 'C − B was −0.79 against G₃ ([−1.00, −0.58], p = 2.1 × 10⁻¹¹)',
       'C − B was −0.92 against G₃ ([−1.25, −0.61], p = 1.2 × 10⁻⁷)'),
 (184, 'the architecture range widens from 1.22 points on validation to 2.59 on test',
       'the architecture range widens from 0.96 points on validation to 1.87 on test'),
 (184, 'is narrow on the test set (3.24 versus 2.59)',
       'is narrow on the test set (3.12 versus 1.87)'),
 (194, 'the augmented arm scores 5.23 points below arm B on validation and 7.01 on test',
       'the augmented arm scores 5.34 points below arm B on validation and 6.10 on test'),
 (209, 'C minus E is −0.18, +0.01 and −0.35 points at V ≥ 1, 2 and 3, against a 1.15-point noise floor — and at the strictest reference the geometric surrogate is better by 2.32 points',
       'C minus E is −0.32, −0.09 and −0.52 points at V ≥ 1, 2 and 3, against a 1.19-point noise floor — and at the strictest reference the geometric surrogate is better by 1.95 points'),
 (214, "against G₂ of 0.8713, statistically indistinguishable from arm B's 0.8719",
       "against G₂ of 0.8675, statistically indistinguishable from arm B's 0.8740"),
 (224, '22.79 against 10.77 points at a fixed threshold of 0.5, and 13.52 against 9.40 at per-arm optimal thresholds',
       '23.07 against 10.64 points at a fixed threshold of 0.5, and 14.22 against 9.70 at per-arm optimal thresholds'),
 (224, 'falls to 0.780 Dice against the unanimous V ≥ 7 reference where a consensus-trained model retains 0.874',
       'falls to 0.771 Dice against the unanimous V ≥ 7 reference where a consensus-trained model retains 0.868'),
 (226, 'score 0.18, 0.50 and 0.87, and the protocol-to-supervision ratio at optimal thresholds is 2.69×, 1.44× and 0.78× respectively',
       'score 0.33, 0.49 and 0.87, and the protocol-to-supervision ratio at optimal thresholds is 6.53×, 1.47× and 0.78× respectively'),
 (234, 'can be reported at 0.6903 or 0.8545 Dice depending only on which consensus level',
       'can be reported at 0.6090 or 0.8648 Dice depending only on which consensus level'),
 (234, 'values the largest architectural change measured here at 1.22 points across an 8-fold parameter range, and the largest difference among five defensible supervision strategies at 2.99.',
       'values the largest architectural change measured here at 0.96 points across an 8-fold parameter range, and the largest difference among five defensible supervision strategies at 2.65.'),
 (237, 'the four supervision targets of this study span 2.99 points',
       'the four supervision targets of this study span 2.65 points'),
 (237, 'Against the 16.42-point consensus-level effect, the 10.02-point threshold effect',
       'Against the 25.58-point consensus-level effect, the 9.42-point threshold effect'),
 (238, 'the 0.8686 obtained by arm D against G₂ is a patch-level, oracle-localised, validation-set value at a threshold of 0.25',
       'the 0.8746 obtained by arm D against G₂ is a patch-level, oracle-localised, validation-set value at a threshold of 0.50'),
 (251, 'move Dice by more than the 1.22-point range measured across five architectures at V ≥ 2',
       'move Dice by more than the 0.96-point range measured across five architectures at V ≥ 2'),
 (252, 'Measured effect: 16.42 points.', 'Measured effect: 25.58 points.'),
 (253, 'Measured effect: 10.02 points.', 'Measured effect: 9.42 points.'),
 (257, 'Measured floor: 1.15 points at 2σ, 95% CI [0.92, 1.52].',
       'Measured floor: 1.19 points at 2σ, 95% CI [0.97, 1.59].'),
 (264, 'and the agreement index (0.18, 0.50, 0.87) cannot be separated',
       'and the agreement index (0.33, 0.49, 0.87) cannot be separated'),
 (266, 'so the reported 1.22-point range is an upper bound',
       'so the reported 0.96-point range is an upper bound'),
 (266, 'The range also widens to 2.59 points on the held-out test set',
       'The range also widens to 1.87 points on the held-out test set'),
 (275, 'Changing only the consensus level used as the reference costs 16.42 points, and the binarisation threshold alone 10.02.',
       'Changing only the consensus level used as the reference costs 25.58 points, and the binarisation threshold alone 9.42.'),
 (276, 'supervision strategies is 3.24 points, and the range across five architectures spanning convolutional and transformer families and an 8-fold parameter range is 2.59 points, of which the four ImageNet-pretrained encoders account for 0.27 — below the 1.15-point noise floor',
       'supervision strategies is 3.12 points, and the range across five architectures spanning convolutional and transformer families and an 8-fold parameter range is 1.87 points, of which the four ImageNet-pretrained encoders account for 0.28 — below the 1.19-point noise floor'),
 (276, 'exceed learning-design choices by 7.0× and the architecture range by 8.7×',
       'exceed learning-design choices by 7.2× and the architecture range by 12.0×'),
]


def run_replace(par, old, new, tag):
    for r in par.runs:
        if old in r.text:
            r.text = r.text.replace(old, new)
            return
    # 跨 run 的情形：合并到首个 run（保留其格式）
    joined = ''.join(r.text for r in par.runs)
    assert old in joined, f'{tag}: 段落中找不到目标文本'
    joined = joined.replace(old, new)
    par.runs[0].text = joined
    for r in par.runs[1:]:
        r.text = ''


def main():
    doc = docx.Document(SRC)
    ps = doc.paragraphs
    for idx, old, new in EDITS:
        run_replace(ps[idx], old, new, f'段落 {idx}')
    doc.save(DST)

    doc2 = docx.Document(DST)
    txt = '\n'.join(p.text for p in doc2.paragraphs)
    for idx, old, new in EDITS:
        assert new in txt, f'回读失败：段落 {idx} 新文本缺失'
        assert old not in txt, f'回读失败：段落 {idx} 旧文本仍在'
    # 摘要词数自检
    a = [i for i, p in enumerate(doc2.paragraphs) if p.text.strip() == 'Abstract'][0]
    h = [i for i, p in enumerate(doc2.paragraphs) if p.text.strip() == 'Highlights'][0]
    n = sum(len(p.text.split()) for p in doc2.paragraphs[a + 1:h]
            if p.text.strip() and not p.text.strip().startswith('Keywords'))
    print(f'摘要词数 {n} (上限 250)')
    assert n <= 250
    print(f'第 1 遍完成：{len(EDITS)} 处正文替换，回读自检通过')
    print(DST)


if __name__ == '__main__':
    main()
