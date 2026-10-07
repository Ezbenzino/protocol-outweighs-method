# -*- coding: utf-8 -*-
"""v16b：在 3.5 节末尾补"测试集 = 五折集成"声明（Codex release-chain 项 5）。

- 备份 v16 docx → 论文_协议效应_v16_before_fold_ensemble.docx
- 在 3.6 标题段（含 "3.6 Patch sampling"）前插入新段落，样式复制自 3.5 节正文段
- 保存回 论文_协议效应_v16_CIBM.docx
"""
import os
import shutil

import docx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'outputs', 'paper', '论文_协议效应_v16_CIBM.docx')
BAK = os.path.join(ROOT, 'outputs', 'paper', '论文_协议效应_v16_before_fold_ensemble.docx')

NEW_TEXT = ('On the held-out test set, all reported numbers are computed from the five fold models '
            'jointly: each fold model predicts the test set, and per-case Dice is averaged across the '
            'five folds before the case-level analyses of Section 3.9. The per-case records therefore '
            'represent a five-fold ensemble; single-fold records are released alongside.')

shutil.copy2(SRC, BAK)
d = docx.Document(SRC)
ps = d.paragraphs

# 找 3.6 标题段
anchor = None
for i, p in enumerate(ps):
    if p.text.strip().startswith('3.6 Patch sampling'):
        anchor = i
        break
if anchor is None:
    raise SystemExit('未找到 3.6 标题段，中止。')

style_par = None
for j in range(anchor - 1, -1, -1):
    t = ps[j].text.strip()
    if t and not t.startswith('3.') and not t.startswith('•') and not t.startswith('Because'):
        style_par = ps[j]
        break
if style_par is None:
    style_par = ps[anchor - 1]

new_p = ps[anchor].insert_paragraph_before()
new_p.style = d.styles[style_par.style.name] if style_par.style is not None else new_p.style
run = new_p.add_run(NEW_TEXT)
# 复制首个 run 的字体格式（若有）
if style_par.runs:
    sr = style_par.runs[0]
    try:
        run.font.name = sr.font.name
        run.font.size = sr.font.size
        if sr.font.bold is not None:
            run.font.bold = sr.font.bold
        if sr.font.italic is not None:
            run.font.italic = sr.font.italic
    except Exception:
        pass

d.save(SRC)
print(f'已插入声明段于 para {anchor} 之前，保存回 {SRC}')
print(f'备份: {BAK}')
print(f'声明: {NEW_TEXT}')
