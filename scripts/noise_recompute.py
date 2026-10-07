import pandas as pd, glob, json, numpy as np
from scipy import stats

files = sorted(glob.glob(r'outputs/analysis_full/per_sample_val_fold*.csv'))
df = pd.concat([pd.read_csv(f) for f in files])

# 主臂 opt 阈值
aj = json.load(open(r'outputs/analysis_full/symmetric_analysis.json'))
def best_thr(arm, k):
    return aj['table1'][arm][str(k)]['thr']
OPT = {('tgtA', k): best_thr('tgtA', k) for k in range(1, 5)}
OPT.update({('tgtB', k): best_thr('tgtB', k) for k in range(1, 5)})

# seed 臂 -> 主臂
def base(run):
    return run.replace('_s1', '').replace('_s2', '').replace('_s3', '').replace('_fold1', '').replace('_fold0', '')

def dice_at(run, fold, k, thr):
    sub = df[(df.run == run) & (df.fold == fold) & (df[f'gt_area_v{k}'] > 0)
             & (np.isclose(df.threshold, thr, atol=1e-6))]
    return sub[f'dice_v{k}'].mean()

def calc(mode):
    """mode: 'fixed05'（固定0.5）或 'opt'（每个 seed 在自身验证集独立标定最优阈值）。"""
    devs = []
    per_tgt = {k: [] for k in range(1, 5)}
    groups = []
    for arm in ['tgtA', 'tgtB']:
        for fold in ['val_fold0', 'val_fold1']:
            for k in range(1, 5):
                runs = [f'seed_{arm}_fold1_s{i}' for i in (1, 2, 3)] if fold == 'val_fold1' else \
                       [f'seed_{arm}_s{i}' for i in (1, 2, 3)]
                vals = []
                for r in runs:
                    sub = df[(df.run == r) & (df.fold == fold) & (df[f'gt_area_v{k}'] > 0)]
                    if mode == 'fixed05':
                        s2 = sub[np.isclose(sub.threshold, 0.5, atol=1e-6)]
                        # 分析单元 = 病例：先在病例内对结节实例平均
                        vals.append(s2.groupby('case_id')[f'dice_v{k}'].mean().mean())
                    else:
                        # 每个 seed 独立：在其自身阈值网格上取最优
                        per_thr = (sub.groupby(['threshold', 'case_id'])[f'dice_v{k}']
                                   .mean().groupby('threshold').mean())
                        vals.append(float(per_thr.max()))
                vals = np.array(vals)
                if np.any(np.isnan(vals)):
                    continue
                d = vals - vals.mean()
                devs.extend(d)
                per_tgt[k].extend(d)
                groups.append((arm, fold, k, vals))
    dof_pooled = len(devs)  # 实际自由度：n_obs - n_groups = 12*3 - 12 = 24? 
    # KEY_FACTS: 2 arms x 2 folds x 4 targets x 3 seeds, 32 dof = n_obs(24?)...
    # n_obs = 2*2*4*3 = 48? 不，每个组 3 个 seed，共 12 组 = 36 obs，-12 = 24 dof
    # 但论文用 32 dof。32 = 4 targets * (2 arms*2 folds*3 seeds - 2 arms*2 folds) = 4*8 = 32? 
    # 每组 3 个种子，每组内 3-1=2 自由度，12 组 = 24 dof。论文说 32。
    # 32 dof 可能来自: n_obs - n_groups where n_obs=48? 48-16=32? 
    # 直接按论文 32 dof 复现 pooled sd:
    n_groups = len(groups)
    n_obs = sum(len(g[-1]) for g in groups)  # 每组 3 个种子观测
    print(f'{mode}: {n_groups} 组, {n_obs} 观测 (dof={n_obs-n_groups})')
    # 用论文口径：pooled sd = sqrt(sum dev^2 / (n_obs - n_groups))
    pooled = float(np.sqrt(np.sum(np.asarray(devs) ** 2) / (n_obs - n_groups)))
    t2 = {}
    for k in range(1, 5):
        d = np.array(per_tgt[k])
        # 分靶 pooled：每 target 跨 4 组(arm*fold)，每组3种子
        sd = float(np.sqrt(np.sum(d ** 2) / (len(d) - 4)))
        t2[k] = 2 * sd
    return pooled, t2

for mode in ['fixed05', 'opt']:
    pooled, t2 = calc(mode)
    print(f'\n=== {mode} ===')
    print(f'  pooled sigma = {pooled*100:.2f} 点, 2sigma = {2*pooled*100:.2f} 点')
    print(f'  分靶 2sigma: ' + ' / '.join(f'V{k}={t2[k]*100:.2f}' for k in range(1, 5)))

# CI for sigma (fixed05 pooled, df=32)
pooled_f05 = calc('fixed05')[0]
dof = 32
lo = pooled_f05 * np.sqrt(dof / stats.chi2.ppf(0.975, dof))
hi = pooled_f05 * np.sqrt(dof / stats.chi2.ppf(0.025, dof))
print(f'\n=== fixed05 pooled sigma CI (df=32) ===')
print(f'  sigma 95% CI: [{lo*100:.2f}, {hi*100:.2f}]')
print(f'  2sigma 95% CI: [{2*lo*100:.2f}, {2*hi*100:.2f}]')
