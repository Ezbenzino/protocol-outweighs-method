# -*- coding: utf-8 -*-
"""qubiq_agreement_index.py —— 直接从 QUBIQ 原始标注计算 rater agreement index。

为什么单独写这个脚本
--------------------
论文 §3.8 的 rater agreement index = Σ|V≥N| / Σ|V≥1|。现行实现
`compute_disagreement.py` 从**评测产生的** per_sample CSV 里取 gt_area，
因此 (a) 必须先训练出模型才能算，(b) 只覆盖跑过模型的数据集。

但这个量只是**标注本身的性质，与模型无关**。直接从原始逐标注者掩膜算，
可以在不训练任何模型的情况下覆盖 QUBIQ 全部 9 个任务，
把论文 §4.14 的"一致性 → 协议主导程度"从 3 个点扩到 10 个点（含 LIDC）。

⚠️ 病例集合口径
---------------
本脚本默认对**全部训练病例**计算；论文现行的 brain/kidney 数值是在
**交叉验证池**上算的，两者会差一点（实测 brain 0.5011 vs 0.4891，
kidney 0.8620 vs 0.8677），差别来自病例集合而非定义。
扩表时必须统一：要么全部用本脚本口径，要么都限定到 CV 池（--cases）。

用法
----
    python scripts/qubiq_agreement_index.py \
        --raw-dir data/raw/qubiq/extracted/training_data_v3_QC

依赖
----
优先用 nibabel；本机没有时回落到文件末尾的最小 NIfTI-1 读取器
（只解析形状与 dtype，对"数掩膜像素"足够；已与 nibabel 路径下的
npz 结果交叉验证：kidney 0.8620 完全一致）。
"""
import argparse
import glob
import gzip
import json
import os
import struct

import numpy as np

try:
    import nibabel as _nib
except ImportError:
    _nib = None

# (子数据集, 任务, 标注者数, 解剖, 模态, 相对 raw-dir 的病例父目录)
TASKS = [
    ('brain-growth',      'task01', 7, 'Brain growth',      'MRI', 'brain-growth/Training'),
    ('brain-tumor',       'task01', 3, 'Brain tumour 1',    'MRI', 'brain-tumor/Training'),
    ('brain-tumor',       'task02', 3, 'Brain tumour 2',    'MRI', 'brain-tumor/Training'),
    ('brain-tumor',       'task03', 3, 'Brain tumour 3',    'MRI', 'brain-tumor/Training'),
    ('kidney',            'task01', 3, 'Kidney',            'CT',  'kidney/Training'),
    ('pancreas',          'task01', 2, 'Pancreas',          'CT',  'pancreas'),
    ('pancreatic-lesion', 'task01', 2, 'Pancreatic lesion', 'CT',  'pancreatic-lesion'),
    ('prostate',          'task01', 6, 'Prostate 1',        'MRI', 'prostate/Training'),
    ('prostate',          'task02', 6, 'Prostate 2',        'MRI', 'prostate/Training'),
]

_DT = {2: np.uint8, 4: np.int16, 8: np.int32, 16: np.float32, 64: np.float64,
       256: np.int8, 512: np.uint16, 768: np.uint32, 1024: np.int64, 1280: np.uint64}


def _read_nii_min(path):
    """最小 NIfTI-1 读取（无 nibabel 时的回落）。"""
    op = gzip.open if path.endswith('.gz') else open
    with op(path, 'rb') as f:
        buf = f.read()
    for endian in ('<', '>'):
        if struct.unpack(endian + 'i', buf[0:4])[0] == 348:
            break
    else:
        raise ValueError(f'{path}: 不是 NIfTI-1')
    dim = struct.unpack(endian + '8h', buf[40:56])
    datatype = struct.unpack(endian + 'h', buf[70:72])[0]
    vox_offset = int(struct.unpack(endian + 'f', buf[108:112])[0]) or 352
    if datatype not in _DT:
        raise ValueError(f'{path}: 不支持的 datatype {datatype}')
    shape = tuple(int(x) for x in dim[1:dim[0] + 1])
    dt = np.dtype(_DT[datatype]).newbyteorder(endian)
    a = np.frombuffer(buf, dtype=dt, count=int(np.prod(shape)), offset=vox_offset)
    return a.reshape(shape[::-1]).astype(np.float32)


def load_mask(path):
    if _nib is not None:
        return (np.asanyarray(_nib.load(path).dataobj) > 0.5).astype(np.int16)
    return (_read_nii_min(path) > 0.5).astype(np.int16)


def index_for_task(raw_dir, rel, task, n_raters, keep=None):
    root = os.path.join(raw_dir, rel)
    s1 = sN = 0
    per, skipped = [], []
    for cd in sorted(d for d in glob.glob(os.path.join(root, '*')) if os.path.isdir(d)):
        cid = os.path.basename(cd)
        if keep is not None and cid not in keep:
            continue
        segs = sorted(glob.glob(os.path.join(cd, f'{task}_seg*.nii*')))
        if len(segs) != n_raters:
            skipped.append([cid, f'{len(segs)} raters'])
            continue
        v = None
        for s in segs:
            m = load_mask(s)
            v = m if v is None else v + m
        a1 = int((v >= 1).sum())
        if a1 == 0:
            skipped.append([cid, 'empty union'])
            continue
        s1 += a1
        sN += int((v >= n_raters).sum())
        per.append(int((v >= n_raters).sum()) / a1)
    if not per:
        return None
    return {'n_raters': n_raters, 'n_cases': len(per),
            'area_mean_ratio': round(s1 and sN / s1, 4),
            'per_case_mean': round(float(np.mean(per)), 4),
            'no_consensus_frac': round(float(np.mean([p == 0 for p in per])), 4),
            'skipped': skipped}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw-dir', default='data/raw/qubiq/extracted/training_data_v3_QC')
    ap.add_argument('--out', default='outputs/analysis/qubiq_agreement_index.json')
    ap.add_argument('--cases', default=None, help='逗号分隔的 case_id csv，用于限定病例集合')
    args = ap.parse_args()

    keep = None
    if args.cases:
        keep = set()
        for f in args.cases.split(','):
            for line in open(f.strip(), encoding='utf-8'):
                t = line.strip()
                if t and t != 'case_id':
                    keep.add(t)

    out = {'definition': 'Σ|V>=N| / Σ|V>=1|，直接由逐标注者掩膜计算，与模型无关',
           'reader': 'nibabel' if _nib is not None else 'built-in minimal NIfTI-1 reader',
           'case_scope': 'CV 池（--cases）' if keep else '全部训练病例',
           'tasks': {}}
    print(f"{'task':26s}{'N':>3}{'cases':>7}{'index':>9}{'no-cons':>9}")
    print('-' * 56)
    for sub, task, n, anat, mod, rel in TASKS:
        r = index_for_task(args.raw_dir, rel, task, n, keep)
        if r is None:
            print(f'{sub}/{task}: 无可用病例，跳过')
            continue
        r.update({'anatomy': anat, 'modality': mod, 'subdataset': sub, 'task': task})
        out['tasks'][f'{sub}/{task}'] = r
        print(f"{sub+'/'+task:26s}{n:>3}{r['n_cases']:>7}"
              f"{r['area_mean_ratio']:>9.4f}{r['no_consensus_frac']:>9.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print('\n已写:', args.out)


if __name__ == '__main__':
    main()
