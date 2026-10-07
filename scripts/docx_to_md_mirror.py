"""docx -> markdown 只读镜像（供 grep/diff）。docx 仍是唯一权威源。
用法: python scripts/docx_to_md_mirror.py [src_docx] [out_md]
默认: outputs/paper/论文初稿_协议效应_v9.docx -> outputs/paper/paper_v9.md
版本号从 src 文件名自动提取（v9 -> "v9"）。

[2026-09-04 停用] 此镜像脚本已停用。历史上出现过 md -> docx 反向构建导致
论文丢图、丢斜体的 62KB 无图 docx 事故（见 docs/项目终审_20260903.md §5）。
为保证 docx 始终是唯一权威源、杜绝"从 md 重建"路径，本脚本物理停用：
任何调用直接退出。paper_draft_v*.md 为历史快照，仅供 grep/diff，不再回灌更新。
"""
import docx, re, sys, os
sys.exit("镜像脚本已停用：docx 是唯一权威源，禁止从 md 重建。见 docstring 说明。")

src = r'outputs/paper/论文初稿_协议效应_v9.docx'
out = r'outputs/paper/paper_v9.md'
if len(sys.argv) >= 3:
    src, out = sys.argv[1], sys.argv[2]

ver = 'v10'
m = re.search(r'[vV](\d+)', os.path.basename(src))
if m:
    ver = 'v' + m.group(1)

doc = docx.Document(src)

def style_level(p):
    try:
        name = (p.style.name or '').lower()
    except AttributeError:
        name = ''
    if 'heading 1' in name: return 1
    if 'heading 2' in name: return 2
    if 'heading 3' in name: return 3
    if 'heading' in name:
        m = re.search(r'(\d+)', name)
        return int(m.group(1)) if m else 2
    return 0

def esc(t):
    # 转义 md 特殊字符（保留中文引号）
    return t

lines = []
table_idx = 0
para_idx = 0

# 需要同时处理段落与表格，保持文档顺序
from docx.document import Document as _Doc
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.oxml.ns import qn

def iter_block_items(parent):
    for child in parent.element.body.iterchildren():
        if child.tag == qn('w:p'):
            yield Paragraph(child, parent)
        elif child.tag == qn('w:tbl'):
            yield Table(child, parent)

for block in iter_block_items(doc):
    if isinstance(block, Paragraph):
        p = block
        t = p.text.strip()
        if not t:
            continue
        lvl = style_level(p)
        if lvl:
            lines.append('#' * lvl + ' ' + t)
        else:
            lines.append(t)
        lines.append('')
    elif isinstance(block, Table):
        lines.append('<!-- TABLE -->')
        for ri, row in enumerate(block.rows):
            cells = [c.text.strip().replace('|', '\\|') for c in row.cells]
            lines.append('| ' + ' | '.join(cells) + ' |')
            if ri == 0:
                lines.append('|' + '---|' * len(cells))
        lines.append('')

content = '\n'.join(lines)
with open(out, 'w', encoding='utf-8') as f:
    f.write(f'# 论文 {ver} 只读镜像（自动回灌自 docx，非权威源）\n\n')
    f.write(f'> 来源: {os.path.basename(src)} ({__import__("datetime").date.today()})。\n')
    f.write('> 本文件仅用于 grep/diff；任何修改请直接改 docx（docx 为权威版本）。\n\n')
    f.write(content)
print(f'已生成 {out}，{len(lines)} 行')
