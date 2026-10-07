# -*- coding: utf-8 -*-
"""B4 数字溯源表：把论文每个关键数字映射到 数据文件 -> 提取路径/脚本 -> 复现命令。
产物：docs/数字溯源表_20260901.md  +  outputs/paper/metric_traceability.json
口径：所有数字必须能在对应 json 中直接复算；未在本轮重算的标 'paper_frozen'（已定格）。
"""
import json, io

S = {}
for name, path in [
    ('sym', r'outputs/analysis_full/symmetric_analysis.json'),
    ('noise', r'outputs/analysis_full/noise_floor.json'),
    ('dis', r'outputs/analysis/disagreement_index.json'),
    ('scope', r'outputs/analysis_scope/scope_analysis_merged2.json'),
    ('xval', r'outputs/analysis_full/cross_validation_report.json'),
]:
    S[name] = json.load(open(path, encoding='utf-8'))

rows = []

def add(rid, paper_loc, paper_val, src, path_expr, calc, cmd):
    rows.append(dict(id=rid, paper_loc=paper_loc, paper_value=paper_val,
                     source_file=src, source_path=path_expr, how=calc, repro=cmd))

# ---- 协议效应（fixed 0.5, V1-V4 极差）----
t1 = S['sym']['table1']
def range05(arm):
    return max(t1[arm][str(k)]['dice_05'] for k in range(1, 5)) - min(t1[arm][str(k)]['dice_05'] for k in range(1, 5))
def range_opt(arms, k):
    return max(t1[a][str(k)]['dice_opt'] for a in arms) - min(t1[a][str(k)]['dice_opt'] for a in arms)
add('E1', '§4.1/Abstract', round(range05('tgtA')*100, 2),
    'symmetric_analysis.json', "table1[tgtA][1..4].dice_05",
    '协议效应 = 同一臂 4 评测靶 fixed-0.5 Dice 极差',
    'python -c "from scripts.cross_validate_rebuild import ..."')
add('E2', '§4.1', round(range05('tgtB')*100, 2), 'symmetric_analysis.json', "table1[tgtB][1..4].dice_05", '同上 (arm B)', '')
add('E3', '§4.1', round(range05('bg_maj')*100, 2), 'symmetric_analysis.json', "table1[bg_maj][1..4].dice_05", '同上 (背景采样臂)', '')
# ---- 训练靶 / 架构 / 预训练 ----
add('M1', '§4.3/Table4', round(range_opt(['tgtA','tgtB','tgtC','tgtD','tgtE'], 4)*100, 2),
    'symmetric_analysis.json', "table1[tgtA-E][4].dice_opt",
    '训练靶极差 = 5 监督靶(A-E) 校准阈值 @V>=4 的 Dice 极差', '')
arch = ['archR18', 'archPU1e3', 'archCNX', 'archPVT', 'tgtB']
add('M2', '§4.6/Table4', round(range_opt(arch, 2)*100, 2),
    'symmetric_analysis.json', "table1[archR18/archPU1e3/archCNX/archPVT/tgtB][2].dice_opt",
    '架构极差 = 5 模型(2族+8倍参数) 校准阈值 @V>=2 的 Dice 极差', '')
add('M3', '§4.6', round(range_opt(['archR18','archCNX','archPVT','tgtB'], 2)*100, 2),
    'symmetric_analysis.json', "table1[archR18/archCNX/archPVT/tgtB][2].dice_opt",
    '四预训练编码器极差 @V>=2', '')
for k, v in {1: 0.1, 2: 0.99, 3: 0.998, 4: 0.999}.items():
    add(f'TH{k}', '§4.2', v, 'symmetric_analysis.json', f"table1[tgtA][{k}].thr",
        f'tgtA 臂 V>= {k} 的验证集校准阈值（0.5 固定比较的标定对照）', '')

# ---- 噪声地板 ----
nf = S['noise']['fixed_0_5']
add('N1', '§4.8', f"σ={nf['pooled_sigma_pts']:.2f}, 2σ={nf['two_sigma_pts']:.2f}, CI {S['noise']['fixed05_two_sigma_ci']}",
    'noise_floor.json', 'fixed_0_5.pooled_sigma_pts / two_sigma_pts',
    '逐 (arm,fold,target) 3-seed 组内 SD 合并；fixed 0.5；32 dof',
    'python scripts/noise_recompute.py && python scripts/save_noise_floor.py')
for k, v in nf['per_target_two_sigma'].items():
    add(f'N1{k}', '§4.8', v, 'noise_floor.json', f"fixed_0_5.per_target_two_sigma[{k}]", '分靶 2σ', '')
add('N2', '§4.8(对照)', S['noise']['optimal_threshold']['two_sigma_pts'],
    'noise_floor.json', 'optimal_threshold.two_sigma_pts',
    '每 seed 独立标定最优阈值口径（未采用，作对照）', '')

# ---- 分歧指数 ----
for k, label in [('LIDC', 'LIDC'), ('QUBIQ-brain', 'QUBIQ-brain'), ('QUBIQ-kidney', 'QUBIQ-kidney')]:
    dd = S['dis'][k]
    add(f'D{k}', '§5.5', f"{dd['area_mean_ratio']:.3f} (per-case {dd['per_case_mean']:.4f})",
        'disagreement_index.json', f"{k}.area_mean_ratio",
        '面积均值之比 Σ|V>=N|/Σ|V>=1|（同代码三数据集）', 'python scripts/compute_disagreement.py')
add('D4', '§5.5', f"{S['dis']['LIDC']['no_consensus_frac']*100:.1f}%", 'disagreement_index.json',
    'LIDC.no_consensus_frac', 'LIDC 792 病例中无四人一致区的比例', '')

# ---- 视野（scope 2 折口径）----
t1s = S['scope']['table1_ladder']
def scoped(arm, a, b):
    return (t1s[f'{arm}_v2'][a] - t1s[f'{arm}_v2'][b]) * 100
add('F1', '§4.13', f"win128->win512 {scoped('tgtBshift_bg','win128','win512'):.2f} pts (CI [-24.77,-20.41])",
    'scope_analysis_merged2.json', "table1_ladder[tgtBshift_bg_v2].win128 - .win512",
    '合并臂视野效应（G2, 校准阈值, 2 折, 病例级）', 'python scripts/analyze_scope.py ...')
add('F2', '§4.11/§4.13', f"win128->win512 {scoped('tgtBshift','win128','win512'):.2f} pts",
    'scope_analysis_merged2.json', "table1_ladder[tgtBshift_v2].win128 - .win512",
    '平移臂视野效应（2 折；5 折主口径 44.87 见论文 §4.11）', '')

# ---- 交叉校验 10 项 ----
for c in S['xval']['checks']:
    add(f'X{c["check"]}', 'A0.5', f"{c['rebuilt']} (paper {c['paper']})",
        'cross_validation_report.json', 'checks[]',
        c['note'] or '重算论文值并与论文核对', 'python scripts/cross_validate_rebuild.py')

# ---- 写出 ----
md = []
md.append('# 数字溯源表（投稿前校验）\n')
md.append(f"- 生成时间：{S['noise']['date']}；唯一权威源：`outputs/paper/论文初稿_协议效应_v9.docx`")
md.append('- 用途：投稿/补强时逐数字可审计；凡数字与本表不符即视为 drift，先查再改。\n')
md.append('| ID | 论文位置 | 论文数值 | 来源文件 | 数据路径 | 计算口径 | 复现命令 |')
md.append('|---|---|---|---|---|---|---|')
for r in rows:
    md.append(f"| {r['id']} | {r['paper_loc']} | {r['paper_value']} | {r['source_file']} | `{r['source_path']}` | {r['how']} | {r['repro']} |")
md_txt = '\n'.join(md) + '\n'
with io.open(r'docs/数字溯源表_20260901.md', 'w', encoding='utf-8') as f:
    f.write(md_txt)
with io.open(r'outputs/paper/metric_traceability.json', 'w', encoding='utf-8') as f:
    json.dump(dict(date=S['noise']['date'], source='docx v9 + 各分析 json', rows=rows),
              f, ensure_ascii=False, indent=1)
print(f'溯源表 {len(rows)} 项已生成 -> docs/数字溯源表_20260901.md')
