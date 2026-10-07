# -*- coding: utf-8 -*-
"""从逐样本 CSV 中删除某个 run 前缀的全部行（流式处理，适合 500 MB 级文件）。
用法：python scripts/drop_runs_from_csv.py <csv路径> <run前缀>
例：  python scripts/drop_runs_from_csv.py outputs/analysis_test/per_sample_test.csv archPU_fold
只删 run 列以该前缀开头的行（例如 archPU_fold0..4），不会误删 archPU1e3_fold*。
先写临时文件再原子替换；请事先备份（run_fix_testset_plainunet_20261002.ps1 已自动备份）。"""
import csv
import os
import sys

path, prefix = sys.argv[1], sys.argv[2]
tmp = path + '.tmp'
kept = dropped = 0
csv.field_size_limit(10 ** 9)
with open(path, newline='', encoding='utf-8') as fi, open(tmp, 'w', newline='', encoding='utf-8') as fo:
    r = csv.reader(fi)
    w = csv.writer(fo)
    header = next(r)
    w.writerow(header)
    i = header.index('run')
    for row in r:
        if row[i].startswith(prefix):
            dropped += 1
        else:
            w.writerow(row)
            kept += 1
os.replace(tmp, path)
print(f'{os.path.basename(path)}: dropped {dropped} rows with run prefix "{prefix}", kept {kept}')
