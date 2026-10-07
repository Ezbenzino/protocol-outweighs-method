# -*- coding: utf-8 -*-
"""v16 final polish（可复现）：两项修订，从插入前备份恢复后执行。

1. para(220) 比值修复：22.51 视野效应相对监督靶/架构的比值
   "7.0 times ... 8.7 times" → "7.2 times ... 12.0 times"
   （v16 更新分子 22.51 时漏改分母：22.51/3.12=7.21, 22.51/1.87=12.04；
    旧值 7.0/8.7 = 22.51/3.24 与 22.51/2.59，即 v14 分母）
2. 在 3.5 节末尾（3.6 标题前）插入"测试集 = 五折集成"声明
"""
import os
import shutil

import docx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 原文件被 WPS 占用（pid 1904），本脚本输出到 _updated 新文件，避免覆盖用户正在打开的文档
BAK = os.path.join(ROOT, 'outputs', 'paper', '论文_协议效应_v16_before_fold_ensemble.docx')
SRC = os.path.join(ROOT, 'outputs', 'paper', '论文_协议效应_v16_CIBM_updated.docx')

ENSEMBLE_TEXT = ('On the held-out test set, all reported numbers are computed from the five fold models '
                 'jointly: each fold model predicts the test set, and per-case Dice is averaged across the '
                 'five folds before the case-level analyses of Section 3.9. The per-case records therefore '
                 'represent a five-fold ensemble; single-fold records are released alongside.')

# 0) 从备份（v16 原始）开始，保证可复现
if os.path.exists(BAK):
    shutil.copy2(BAK, SRC)
    print('已从备份生成起点文件')
else:
    print('无备份，直接修改当前文件')

d = docx.Document(SRC)
ps = d.paragraphs

# 1) 比值修复（文本锚点定位）
fixed = 0
for p in ps:
    if 'That is still roughly twenty times the run-to-run noise floor' in p.text and '7.0 times' in p.text:
        for r in p.runs:
            if '7.0 times' in r.text:
                r.text = r.text.replace('7.0 times', '7.2 times')
            if '8.7 times' in r.text:
                r.text = r.text.replace('8.7 times', '12.0 times')
        fixed += 1
        print('已修复 para:', p.text[:130], '...')
if fixed == 0:
    print('!! 未找到待修复的 7.0/8.7 段落，请人工检查')
else:
    print(f'比值修复: {fixed} 段')

# 2) 五折集成声明（锚点：3.6 标题段前插入）
anchor = None
for i, p in enumerate(ps):
    if p.text.strip().startswith('3.6 Patch sampling'):
        anchor = i
        break
if anchor is None:
    print('!! 未找到 3.6 标题段，跳过声明插入')
else:
    style_par = None
    for j in range(anchor - 1, -1, -1):
        t = ps[j].text.strip()
        if t and not t.startswith('3.') and not t.startswith('•'):
            style_par = ps[j]
            break
    if style_par is None:
        style_par = ps[anchor - 1]
    new_p = ps[anchor].insert_paragraph_before()
    new_p.style = d.styles[style_par.style.name] if style_par.style is not None else new_p.style
    run = new_p.add_run(ENSEMBLE_TEXT)
    if style_par.runs:
        sr = style_par.runs[0]
        try:
            run.font.name = sr.font.name
            run.font.size = sr.font.size
        except Exception:
            pass
    print(f'已插入五折集成声明于 para {anchor} 前')

d.save(SRC)
print(f'已保存 {SRC}')
