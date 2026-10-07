# -*- coding: utf-8 -*-
"""在 v17 权威源上应用 QUBIQ 8 点改写（内容匹配定位，段落编号与 v16 有偏移）。"""
import shutil
import docx
from docx.shared import Inches
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = str(ROOT / 'outputs' / 'paper' / '论文_协议效应_v17_CIBM.docx')
BAK = str(ROOT / 'outputs' / 'paper' / '论文_协议效应_v17_CIBM_backup_QUBIQ8.docx')
NEW_IMG = str(ROOT / 'outputs' / 'figures' / 'fig9_generalisation.png')

shutil.copyfile(SRC, BAK)
print('已备份 v17 ->', BAK)

d = docx.Document(SRC)

def find_para(sub, must=True):
    hits = [i for i, p in enumerate(d.paragraphs) if sub in p.text]
    if len(hits) == 1:
        return hits[0]
    if must:
        raise SystemExit('定位失败(命中%d): %r' % (len(hits), sub))
    return None

def set_text(idx, new_text):
    p = d.paragraphs[idx]
    for r in list(p.runs):
        r._element.getparent().remove(r._element)
    p.add_run(new_text)

# 1. §3.8 设计（"was repeated on two tasks of the QUBIQ 2021 collection"）
i = find_para('was repeated on two tasks of the QUBIQ 2021 collection')
set_text(i, (
    'To test whether the ordering of effects is specific to LIDC, the identical controlled design — '
    'three supervision targets (union, majority, consensus-soft), five-fold cross-validation, the '
    'full threshold sweep, and evaluation against every nested reference — was repeated on seven '
    'tasks spanning six collections of the QUBIQ 2021 benchmark [12]: brain-growth (MRI, 7 raters), '
    'kidney (CT, 3 raters), three brain-tumour subtasks (MRI, 3 raters each) and two prostate '
    'subtasks (MRI, 6 raters each). These span two modalities, several anatomies and rater counts '
    'from 3 to 7. Dataset-specific handling is limited to the number of raters, which sets the '
    'number of nested references, and to the batch size, which the small cohorts require. The three '
    'brain-tumour subtasks share the same readers and cases, as do the two prostate subtasks; each '
    'subtask is a distinct annotation task and is treated as one observation point in the trend '
    'analysis, with the shared-reader structure addressed explicitly in Section 4.14.'))
print('已改 §3.8 设计段')

# 2. §4.14 开头（"The controlled design was repeated in full on two QUBIQ 2021 tasks"）
i = find_para('The controlled design was repeated in full on two QUBIQ 2021 tasks')
set_text(i, (
    'The controlled design was repeated in full on seven QUBIQ 2021 tasks spanning six collections '
    '[12] — brain-growth, kidney, three brain-tumour subtasks and two prostate subtasks — which '
    'differ from LIDC in modality, anatomy and rater count.'))
print('已改 §4.14 开头段')

# 3. §4.14 主结论（"The three points are strictly monotone"）
i = find_para('The three points are strictly monotone')
set_text(i, (
    'A single continuous quantity reconciles the datasets. Across the seven QUBIQ tasks the rater '
    'agreement index and the absolute protocol effect are negatively correlated: Spearman rho = '
    '-0.79 (exact permutation p = 0.048, Table 7, Fig. 9); including LIDC as a reference point '
    'yields rho = -0.86 (p = 0.007). The association is robust to dropping any single task (rho '
    'remains between -0.79 and -0.89). Three structural features underline the trend. First, the '
    'absolute protocol effect decays with agreement: the lowest-agreement task (brain-tumour t2, '
    'index 0.18) shows the largest effect (30.4 points at the optimal threshold), while the '
    'highest-agreement task (prostate t1, 0.96) is among the smallest (2.4 points). Second, the '
    'supervision effect collapses toward zero with agreement: at the highest-agreement tasks the '
    'three supervision targets become near-equivalent (0.5-0.8 points), so the choice of training '
    'target becomes almost immaterial when raters largely agree. Third, the ratio measure is '
    'distorted at the high-agreement end: as the supervision denominator tends to zero, small '
    'absolute gaps translate into large ratios (prostate t1, 4.58x), so the ratio should be read '
    'alongside the absolute effect rather than alone.'))
print('已改 §4.14 主结论段')

# 4. §4.14 limits（"And with three datasets the rater count (3, 4 and 7)"）
i = find_para('And with three datasets the rater count (3, 4 and 7)')
set_text(i, (
    'This is a more useful statement than a claim of universal dominance, because it comes with a '
    'predicted boundary: the results of this paper should be expected to apply to tasks with '
    'substantial rater disagreement, and not to tasks on which raters essentially agree. Three '
    'limits must be stated with it. Only the evaluation-target axis was measured externally — the '
    'two largest protocol effects, field of view and position offset, were not re-measured on '
    'QUBIQ. Second, the rater count (3, 3, 3, 6, 6 and 7) is no longer fully confounded with the '
    'agreement index, but the brain-tumour subtasks share readers and cases, as do the two prostate '
    'subtasks, so the seven points are not fully independent; the exact permutation test is one '
    'remedy, and the association should be read as observed rather than causal. Third, the '
    'agreement index is an area-based measure: tasks with area agreement can still differ at the '
    'boundary, so the trend is best interpreted in absolute protocol-effect terms.'))
print('已改 §4.14 limits 段')

# 5. §5 局限（"validated externally on two QUBIQ tasks"）
i = find_para('validated externally on two QUBIQ tasks')
set_text(i, (
    'Single benchmark for three of the four protocol axes. The evaluation-target axis is validated '
    'externally on seven QUBIQ tasks spanning six collections (Section 4.14), but the field-of-view '
    'and position-offset axes — the two largest effects — were measured on LIDC-IDRI only. Their '
    'magnitude on other datasets is unknown.'))
print('已改 §5 局限段')

# 6. brain-growth 段末句限定（"The ordering therefore replicates across dataset, modality, anatomical site and rater count."）
i = find_para('The ordering therefore replicates across dataset, modality, anatomical site and rater count')
p = d.paragraphs[i]
full = p.text.replace(
    'The ordering therefore replicates across dataset, modality, anatomical site and rater count.',
    'The protocol-dominates ordering therefore replicates from LIDC to a low-to-moderate-agreement '
    'MRI task with different anatomy and rater count, and vanishes once agreement is high (kidney below).')
for r in list(p.runs):
    r._element.getparent().remove(r._element)
p.add_run(full)
print('已限定 brain-growth 段末句')

# 7. Fig 9 图片替换 + caption
from docx.oxml.ns import qn
fig_idx = None
for i, p in enumerate(d.paragraphs):
    if p._element.findall('.//' + qn('w:drawing')):
        nxt = d.paragraphs[i + 1].text if i + 1 < len(d.paragraphs) else ''
        if nxt.startswith('Fig. 9'):
            fig_idx = i
            break
assert fig_idx is not None, '未找到 Fig 9 图片'
for r in list(d.paragraphs[fig_idx].runs):
    r._element.getparent().remove(r._element)
run = d.paragraphs[fig_idx].add_run()
run.add_picture(NEW_IMG, width=Inches(6.5))
cap_idx = fig_idx + 1
for r in list(d.paragraphs[cap_idx].runs):
    r._element.getparent().remove(r._element)
d.paragraphs[cap_idx].add_run(
    'Fig. 9 — (a) Absolute protocol effect against the rater agreement index for LIDC and the seven '
    'QUBIQ tasks (Spearman rho = -0.86, permutation p = 0.048). (b) The two component effect sizes '
    'per task at per-arm optimal thresholds.')
print('已替换 Fig 9 图片与 caption')

# 8. Table 7 扩展（tables[8]）
t = d.tables[8]
for r in list(t.rows[3:]):
    r._element.getparent().remove(r._element)
NEW = [
    ('QUBIQ brain-tumour t2', '3', '0.18', '44.58', '23.52', '1.90x', '30.35', '17.83', '1.70x'),
    ('QUBIQ brain-growth',    '7', '0.49', '23.07', '10.64', '2.17x', '14.22', '9.70',  '1.47x'),
    ('QUBIQ prostate t2',     '6', '0.77', '15.12', '2.22',  '6.82x', '14.22', '2.96',  '4.81x'),
    ('QUBIQ brain-tumour t3', '3', '0.64', '6.27',  '5.73',  '1.09x', '2.98',  '2.98',  '1.00x'),
    ('QUBIQ prostate t1',     '6', '0.96', '2.47',  '0.67',  '3.67x', '2.40',  '0.52',  '4.58x'),
    ('QUBIQ brain-tumour t1', '3', '0.78', '2.61',  '1.61',  '1.62x', '2.21',  '1.77',  '1.25x'),
    ('QUBIQ kidney',          '3', '0.87', '3.76',  '2.60',  '1.45x', '0.61',  '0.79',  '0.78x'),
]
for (name, raters, agree, fp, fs, fr, op, osv, ort) in NEW:
    r1 = t.add_row()
    for ci, v in enumerate([name, raters, agree, 'fixed 0.5', fp, fs, fr]):
        r1.cells[ci].text = v
    r2 = t.add_row()
    for ci, v in enumerate(['', '', '', 'optimal', op, osv, ort]):
        r2.cells[ci].text = v
# 比率符号统一
for r in t.rows:
    for c in r.cells:
        for p in c.paragraphs:
            for run in p.runs:
                if run.text.endswith('x') and not run.text.endswith('×'):
                    run.text = run.text[:-1] + '×'
print('已扩展 Table 7 -> %d 行' % len(t.rows))

# 9. Table 7 caption 更新（定位 "Table 7. Protocol"）
i = find_para('Table 7. Protocol (evaluation-target) and supervision-target effects across datasets')
set_text(i, (
    'Table 7. Protocol (evaluation-target) and supervision-target effects for LIDC and the seven '
    'QUBIQ tasks under a fixed threshold of 0.5 and at per-target optimal thresholds. Protocol '
    'effect: the largest across training arms of the gap between Dice at V >= 1 and V >= N at that '
    'arm\u2019s threshold. Supervision effect: the largest across evaluation targets of the '
    'between-arm spread. Agreement index: area-mean ratio |V >= N| / |V >= 1|, larger meaning '
    'greater rater agreement. Tasks are ordered by their protocol effect at optimal thresholds.'))
print('已更新 Table 7 caption')

d.save(SRC)
print('v17 改写完成')
