# -*- coding: utf-8 -*-
"""fix_qubiq_consensus.py —— 修复 brain_tumor_task02/03、prostate_task02 的 consensus 数据。

背景：预处理阶段这三个数据集错用了 task01 的 seg（npz consensus 与 task01 完全一致）。
原始 seg 是不同的（task02/03 语义不同，部分 rater 空标注）。

修复逻辑（已在 case01 逐位验证）：
- 坐标变换：seg 需 .T（transpose，swap y/x）后与 npz consensus 对齐（brain_tumor 与 prostate 均验证通过）。
- brain_tumor：consensus 为 (4,240,240)，标注只在第 0 层；cons[0] = sum(seg_k.T>0)。
- prostate：consensus 为 (1,640,640)；cons[0] = sum(seg_k.T.squeeze()>0)。
- slices（图像）是 case 级，与标注无关，保持不变。

执行后：三个数据集的 npz consensus 变为各自 task 的正确 vote map。
"""
import os
import numpy as np
import nibabel as nib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, 'data', 'raw', 'qubiq', 'extracted', 'training_data_v3_QC')

# 数据集 -> (短名, raw 任务根, 子任务名, n_raters, consensus 层数)
SPECS = [
    ('brain_tumor_task02', 'brain-tumor', 'task02', 3, 4),
    ('brain_tumor_task03', 'brain-tumor', 'task03', 3, 4),
    ('prostate_task02',    'prostate',    'task02', 6, 1),
]


def main():
    for ds, raw_sub, task, n, n_layers in SPECS:
        npz_dir = os.path.join(ROOT, 'data', 'qubiq', ds, 'npz')
        raw_dir = os.path.join(RAW, raw_sub, 'Training')
        cases = sorted(os.listdir(npz_dir))
        changed = 0
        for case in cases:
            if not case.endswith('.npz'):
                continue
            cid = case[:-4]
            npz_path = os.path.join(npz_dir, case)
            z = np.load(npz_path, allow_pickle=True)
            slices = z['slices']
            # 读该 task 的 n 个 rater 掩膜并叠加（缺失的 rater 视为空掩膜，QUBIQ 部分 case 部分 rater 未标注）
            vote = None
            for r in range(1, n + 1):
                seg_path = os.path.join(raw_dir, cid, f'{task}_seg{r:02d}.nii.gz')
                if not os.path.exists(seg_path):
                    continue
                seg = nib.load(seg_path).get_fdata()
                m = (seg > 0).astype(np.uint8)
                if m.ndim == 3:
                    m = m.squeeze()  # (H,W,1) -> (H,W)
                if vote is None:
                    vote = np.zeros_like(m.T, dtype=np.uint8)
                vote += m.T  # transpose 对齐 npz 坐标
            # consensus 形状：brain_tumor 固定 (4,240,240) 且标注在第 0 层；prostate 跟随 vote 尺寸
            if n_layers == 4:
                cons_new = np.zeros((4, 240, 240), dtype=np.uint8)
                cons_new[0] = vote
            else:
                cons_new = np.zeros((1,) + vote.shape, dtype=np.uint8)
                cons_new[0] = vote
            # 保存（保留原 slices）
            np.savez_compressed(npz_path, slices=slices, consensus=cons_new)
            changed += 1
        print(f'{ds}: 重建 {changed} 个 case consensus')


def verify():
    """验证：重建后的 task02/03 与 task01 不同，且 vote 分布合理。"""
    import hashlib
    for ds, raw_sub, task, n, n_layers in SPECS:
        npz_dir = os.path.join(ROOT, 'data', 'qubiq', ds, 'npz')
        ref_dir = os.path.join(ROOT, 'data', 'qubiq',
                               ds.replace('task02', 'task01').replace('task03', 'task01'), 'npz')
        same = 0; total = 0
        for fn in sorted(os.listdir(npz_dir))[:10]:
            if not fn.endswith('.npz'):
                continue
            a = np.load(os.path.join(npz_dir, fn), allow_pickle=True)['consensus']
            b = np.load(os.path.join(ref_dir, fn), allow_pickle=True)['consensus']
            if np.array_equal(a, b):
                same += 1
            total += 1
            print(f'  {ds}/{fn}: max={int(a.max())} 与task01相同? {np.array_equal(a,b)}')
        print(f'{ds}: 与 task01 相同的 case 数 = {same}/{total}（应为 0）')


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--verify':
        verify()
    else:
        main()
