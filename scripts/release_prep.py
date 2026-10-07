# -*- coding: utf-8 -*-
"""release_prep.py —— 发布前扫描：绝对路径 / 敏感信息 / 大文件。
只报告、不修改；人工处理后重跑至全绿。
用法：python scripts/release_prep.py
"""
import os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXCLUDE_DIRS = {'__pycache__', '.pytest_cache', 'outputs', 'data', 'external_repos',
                'refs', '.git', 'docs', 'tmp'}
CODE_EXTS = {'.py', '.yaml', '.yml', '.json', '.md', '.txt', '.sh'}

abspath_re = re.compile(r'[A-Za-z]:[\\/](?!/)|/Users/|/home/')
leak_re = re.compile(r'token|api[_-]?key|password|secret|open_id|user_id|sk-\w{8}', re.I)
abs_count = 0
leak_hits = []

for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
    for fn in filenames:
        ext = os.path.splitext(fn)[1].lower()
        if ext not in CODE_EXTS:
            continue
        p = os.path.join(dirpath, fn)
        rel = os.path.relpath(p, ROOT)
        if rel.replace('\\', '/').endswith('scripts/release_prep.py'):
            continue  # 扫描器自身的正则包含路径模式，跳过
        try:
            with open(p, encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f, 1):
                    if abspath_re.search(line):
                        abs_count += 1
                        print(f'[ABS] {rel}:{i}: {line.strip()[:100]}')
                    if leak_re.search(line):
                        leak_hits.append((rel, i, line.strip()[:80]))
        except Exception as e:
            print(f'[SKIP] {rel}: {e}')

print(f'\n=== 绝对路径命中: {abs_count} 处（发布前需相对化） ===')
print(f'=== 敏感关键词命中: {len(leak_hits)} 处 ===')
for rel, i, s in leak_hits:
    print(f'  {rel}:{i}: {s}')
if abs_count == 0 and not leak_hits:
    print('  全部通过 ✅')
print('\n下一步：按上述报告逐项人工处理；发布前重跑本脚本确认全绿。')
