# -*- coding: utf-8 -*-
"""gen_qubiq_configs.py —— 为 QUBIQ 新增任务生成 A/B/C 三个训练靶配置。

配置只在四处与现有 configs/qubiq/{brain,kidney}_{A,B,C}.yaml 不同：
注释、loss.target、data.n_raters、data.npz_dir / split_dir。
因此直接以 brain_A/B/C 为模板做替换，不新造 schema。
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(ROOT, 'configs', 'qubiq')

# short, 子数据集, 任务, 标注者数, 中文名
NEW = [
    ('btum1',      'brain-tumor',       'task01', 3, 'QUBIQ-brain-tumor 任务 1'),
    ('btum2',      'brain-tumor',       'task02', 3, 'QUBIQ-brain-tumor 任务 2'),
    ('btum3',      'brain-tumor',       'task03', 3, 'QUBIQ-brain-tumor 任务 3'),
    ('panc',       'pancreas',          'task01', 2, 'QUBIQ-pancreas'),
    ('panclesion', 'pancreatic-lesion', 'task01', 2, 'QUBIQ-pancreatic-lesion'),
    ('pros1',      'prostate',          'task01', 6, 'QUBIQ-prostate 任务 1'),
    ('pros2',      'prostate',          'task02', 6, 'QUBIQ-prostate 任务 2'),
]
ARM_CN = {'A': '并集 V>=1', 'B': '多数硬掩膜 V>=2', 'C': '多数硬掩膜 + 软共识 p=V/N'}


def main():
    n = 0
    for short, sub, task, nr, cn in NEW:
        for arm in ('A', 'B', 'C'):
            src = os.path.join(CFG, f'brain_{arm}.yaml')
            s = open(src, encoding='utf-8').read()
            head = (f'# {cn} 训练靶 {arm}：{ARM_CN[arm]}（与 LIDC 同构，n_raters={nr}）。\n'
                    f'# 外部数据集验证扩展（2026-09-04）：把 §4.14 的一致性-协议关系从 3 点扩到 10 点。\n')
            s = re.sub(r'\A(#[^\n]*\n)+', head, s)
            s = s.replace('  n_raters: 7', f'  n_raters: {nr}')
            s = s.replace('  npz_dir: data/qubiq/brain/npz', f'  npz_dir: data/qubiq/{short}/npz')
            s = s.replace('  split_dir: data/qubiq/brain/splits', f'  split_dir: data/qubiq/{short}/splits')
            dst = os.path.join(CFG, f'{short}_{arm}.yaml')
            open(dst, 'w', encoding='utf-8', newline='\n').write(s)
            n += 1
    print(f'已生成 {n} 个配置 -> {CFG}')
    # 自检：每个新配置的 n_raters / 目录都替换成功
    for short, sub, task, nr, cn in NEW:
        for arm in ('A', 'B', 'C'):
            t = open(os.path.join(CFG, f'{short}_{arm}.yaml'), encoding='utf-8').read()
            assert f'n_raters: {nr}' in t, f'{short}_{arm}: n_raters 未替换'
            assert f'data/qubiq/{short}/npz' in t, f'{short}_{arm}: npz_dir 未替换'
            assert 'data/qubiq/brain' not in t, f'{short}_{arm}: 仍残留 brain 路径'
    print('自检通过')


if __name__ == '__main__':
    main()
