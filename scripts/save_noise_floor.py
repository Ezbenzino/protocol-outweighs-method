import pandas as pd, glob, json, numpy as np
from scipy import stats

files = sorted(glob.glob(r'outputs/analysis_full/per_sample_val_fold*.csv'))
df = pd.concat([pd.read_csv(f) for f in files])

def calc(mode):
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
                        per_thr = (sub.groupby(['threshold', 'case_id'])[f'dice_v{k}']
                                   .mean().groupby('threshold').mean())
                        vals.append(float(per_thr.max()))
                vals = np.array(vals)
                if np.any(np.isnan(vals)):
                    continue
                d = vals - vals.mean()
                devs.extend(d)
                per_tgt[k].extend(d)
                groups.append(vals)
    n_obs = sum(len(g) for g in groups)
    n_groups = len(groups)
    dof = n_obs - n_groups
    pooled = float(np.sqrt(np.sum(np.asarray(devs) ** 2) / dof))
    t2 = {}
    for k in range(1, 5):
        d = np.array(per_tgt[k])
        sd = float(np.sqrt(np.sum(d ** 2) / (len(d) - 4)))
        t2[f'V>={k}'] = round(2 * sd * 100, 2)
    return {'pooled_sigma_pts': round(pooled * 100, 2), 'two_sigma_pts': round(2 * pooled * 100, 2),
            'dof': dof, 'per_target_two_sigma': t2}

fixed = calc('fixed05')
opt = calc('opt')

# fixed05 σ 的 CI（df=32）
dof = 32
lo = fixed['pooled_sigma_pts']/100 * np.sqrt(dof / stats.chi2.ppf(0.975, dof))
hi = fixed['pooled_sigma_pts']/100 * np.sqrt(dof / stats.chi2.ppf(0.025, dof))
out = {
    'date': '2026-09-01',
    'source': 'outputs/analysis_full/per_sample_val_fold0-4.csv (P0 重跑, seed_tgtA/B s1-s3, fold0+fold1)',
    'method': '逐 (arm,fold,target) 组 3-seed 组内标准差合并；fixed05=固定阈值0.5；opt=每个seed自身阈值网格最优',
    'fixed_0_5': fixed,
    'optimal_threshold': opt,
    'fixed05_sigma_ci': [round(lo*100, 2), round(hi*100, 2)],
    'fixed05_two_sigma_ci': [round(2*lo*100, 2), round(2*hi*100, 2)],
}
json.dump(out, open(r'outputs/analysis_full/noise_floor.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print(json.dumps(out, ensure_ascii=False, indent=1))
print('\n已保存 outputs/analysis_full/noise_floor.json')
