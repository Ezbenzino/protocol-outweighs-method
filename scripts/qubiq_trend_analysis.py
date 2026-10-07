# QUBIQ 趋势分析：协议效应 vs 一致性指数（Spearman + 精确置换检验）
# 假设：随标注一致性升高，协议效应（相对/绝对）单调下降。
# 输出: outputs/analysis/qubiq_trend.json + 终端表格
# 检验细节：
#   - Spearman 秩相关 rho（基于秩，对单调关系稳健）
#   - 精确置换检验：n 点排列（n! 穷举，n<=9 直接枚举；否则 200k 随机置换）
#   - 不做回归拟合（7-9 点不拟合连续关系，只用秩相关+置换检验）
import glob, os, itertools, json, random
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS = os.path.join(ROOT, 'outputs', 'analysis')

# 与 analyze_qubiq_all.py 相同的 9 点定义（顺序固定，供展示）
DATASETS = [
    dict(key='brain-growth', label='QUBIQ-brain-growth', dir=os.path.join(ANALYSIS, 'qubiq_brain'), n=7),
    dict(key='kidney', label='QUBIQ-kidney', dir=os.path.join(ANALYSIS, 'qubiq_kidney'), n=3),
    dict(key='brain-tumor-task1', label='QUBIQ-brain-tumor-task1', dir=os.path.join(ANALYSIS, 'qubiq_brain_tumor_task01'), n=3),
    dict(key='brain-tumor-task2', label='QUBIQ-brain-tumor-task2', dir=os.path.join(ANALYSIS, 'qubiq_brain_tumor_task02'), n=3),
    dict(key='brain-tumor-task3', label='QUBIQ-brain-tumor-task3', dir=os.path.join(ANALYSIS, 'qubiq_brain_tumor_task03'), n=3),
    dict(key='prostate-task1', label='QUBIQ-prostate-task1', dir=os.path.join(ANALYSIS, 'qubiq_prostate_task01'), n=6),
    dict(key='prostate-task2', label='QUBIQ-prostate-task2', dir=os.path.join(ANALYSIS, 'qubiq_prostate_task02'), n=6),
    dict(key='pancreas', label='QUBIQ-pancreas', dir=os.path.join(ANALYSIS, 'qubiq_pancreas'), n=2),
    dict(key='pancreatic-lesion', label='QUBIQ-pancreatic-lesion', dir=os.path.join(ANALYSIS, 'qubiq_pancreatic_lesion'), n=2),
]


def load_disagreement(cfg):
    """病例级一致性指数：Σ|V>=N| / Σ|V>=1|（GT 面积，与 compute_disagreement.py 同口径）"""
    cols = [f'gt_area_v{k}' for k in range(1, cfg['n'] + 1)]
    parts = []
    for f in sorted(glob.glob(os.path.join(cfg['dir'], 'per_sample_val_fold*.csv'))):
        d = pd.read_csv(f)
        r0 = sorted(d.run.unique())[0]
        t0 = sorted(d.threshold.unique())[0]
        parts.append(d[(d.run == r0) & (d.threshold == t0)][['case_id'] + cols])
    if not parts:
        return None
    inst = pd.concat(parts, ignore_index=True)
    sub = inst.groupby('case_id', as_index=False)[cols].sum()
    return float(sub[f'gt_area_v{cfg["n"]}'].sum() / sub['gt_area_v1'].sum())


def load_protocol(cfg):
    """协议效应（opt_thr 主口径）：max over arms of |Dice@V>=1 - Dice@V>=N|（病例级）"""
    fps = sorted(glob.glob(os.path.join(cfg['dir'], 'per_sample_val_fold*.csv')))
    if not fps:
        return None
    df = pd.concat([pd.read_csv(f) for f in fps], ignore_index=True)
    n = cfg['n']
    cols = [f'dice_v{k}' for k in range(1, n + 1)]
    agg = (df.groupby(['run', 'threshold', 'case_id'])[cols].mean()
             .groupby(['run', 'threshold']).mean().reset_index())
    prefixes = [p for p in ('A', 'B', 'C') if any(agg['run'].str.contains(p))]
    # 用真实 run 前缀：qubiq_<dir名>_<arm>
    run_prefixes = sorted(set(r.split('_fold')[0] for r in agg['run'].unique()))
    proto_max = -1
    sup_max = -1
    sums = {}
    for pre in run_prefixes:
        sub = agg[agg['run'].str.startswith(pre)]
        s = {}
        for k in range(1, n + 1):
            col = f'dice_v{k}'
            s[k] = sub.loc[sub.groupby('run')[col].idxmax(), col].mean()
        sums[pre] = s
        proto_max = max(proto_max, abs(s[1] - s[n]))
    for k in range(1, n + 1):
        vs = [s[k] for s in sums.values()]
        sup_max = max(sup_max, max(vs) - min(vs))
    return {'protocol': proto_max * 100, 'supervision': sup_max * 100,
            'ratio': proto_max / sup_max if sup_max else float('nan')}


def exact_permutation_p(x, y, n_perm=200000, seed=42):
    """Spearman rho 的精确/蒙特卡洛置换 p 值：置换 y 的秩，统计 |rho| >= 观测 |rho| 的比例。"""
    rng = random.Random(seed)
    rx = stats.rankdata(x)
    ry_obs = stats.rankdata(y)
    rho_obs = stats.spearmanr(x, y).statistic
    n = len(x)
    count = 0
    total = 0
    if n <= 9:
        perms = itertools.permutations(range(n))
        for perm in perms:
            ry_p = np.asarray(ry_obs, dtype=float)[list(perm)]
            rho_p = np.corrcoef(rx, ry_p)[0, 1]
            total += 1
            if abs(rho_p) >= abs(rho_obs) - 1e-12:
                count += 1
    else:
        base = np.asarray(ry_obs, dtype=float)
        for _ in range(n_perm):
            ry_p = base[rng.sample(range(n), n)]
            rho_p = np.corrcoef(rx, ry_p)[0, 1]
            total += 1
            if abs(rho_p) >= abs(rho_obs) - 1e-12:
                count += 1
    return rho_obs, count / total, total


def main():
    # 收集 9 点
    rows = []
    for cfg in DATASETS:
        dis = load_disagreement(cfg)
        eff = load_protocol(cfg)
        if dis is None or eff is None:
            print(f"[缺失] {cfg['label']}")
            continue
        rows.append({**cfg, 'disagreement': dis, **eff})

    print('=' * 100)
    print('QUBIQ 任务点汇总（病例级，opt_thr 口径）')
    print('=' * 100)
    print(f"{'数据集':<28}{'N':>3}{'一致性':>10}{'协议效应':>10}{'监督效应':>10}{'比值':>8}")
    for r in rows:
        print(f"{r['label']:<28}{r['n']:>3}{r['disagreement']:>10.3f}{r['protocol']:>10.2f}"
              f"{r['supervision']:>10.2f}{r['ratio']:>8.2f}x")

    n_pts = len(rows)
    if n_pts < 5:
        print(f"\n点数不足（{n_pts}<5），无法做趋势检验")
        return

    x = np.array([r['disagreement'] for r in rows], dtype=float)
    for metric in ['protocol', 'ratio']:
        y = np.array([r[metric] for r in rows], dtype=float)
        if np.isnan(y).any():
            print(f"\n[{metric}] 含 NaN，跳过")
            continue
        rho, p, nperm = exact_permutation_p(x, y)
        rho_sp, p_sp = stats.spearmanr(x, y)
        # 秩相关的方向：负 = 一致性越高效应越小（支持假设）
        direction = '支持假设(负相关)' if rho < 0 else '与假设相反(正相关)'
        print(f"\n[趋势: 一致性指数 vs {metric}]")
        print(f"  Spearman rho = {rho_sp:.3f}  (scipy p = {p_sp:.4f})")
        print(f"  精确置换 p   = {p:.4f}  ({nperm} 次排列)  -> {direction}")

    # 保存
    out = {'n_points': n_pts,
           'rows': [{k: r[k] for k in ('label', 'n', 'modality' if 'modality' in r else 'n')} | 
                    {'disagreement': r['disagreement'], 'protocol': r['protocol'],
                     'supervision': r['supervision'], 'ratio': r['ratio']} for r in rows]}
    for metric in ['protocol', 'ratio']:
        y = np.array([r[metric] for r in rows], dtype=float)
        if np.isnan(y).any():
            continue
        rho, p, nperm = exact_permutation_p(x, y)
        out[f'trend_{metric}'] = {'spearman_rho': rho, 'perm_p': p, 'n_perm': nperm}
    with open(os.path.join(ANALYSIS, 'qubiq_trend.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n已写: {os.path.join(ANALYSIS, 'qubiq_trend.json')}")


if __name__ == '__main__':
    main()
