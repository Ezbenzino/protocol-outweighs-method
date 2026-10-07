# QUBIQ 保留测试集报告：验证折标定最优阈值 -> 导入 test 集
# 与 LIDC 主实验流程对齐（LIDC 也是 val 标定阈值、test 报告）
import pandas as pd, glob, os
import numpy as np

def load(dir_, pattern):
    fps = sorted(glob.glob(os.path.join(dir_, pattern)))
    return pd.concat([pd.read_csv(f) for f in fps], ignore_index=True)

def calibrate_thresholds(val_df, n):
    """对每个 (run, target k) 在 val 折上标定最优阈值"""
    agg = val_df.groupby(['run', 'threshold'])[[f'dice_v{k}' for k in range(1, n + 1)]].mean().reset_index()
    cal = {}
    for run, g in agg.groupby('run'):
        for k in range(1, n + 1):
            row = g.loc[g[f'dice_v{k}'].idxmax()]
            cal[(run, k)] = float(row['threshold'])
    return cal

def report_test(test_df, cal, n):
    """用标定阈值在 test 上取 dice；按 (arm, target) 汇总"""
    rows = []
    for (run, k), thr in cal.items():
        sub = test_df[(test_df['run'] == run) & (np.isclose(test_df['threshold'], thr))]
        if len(sub) == 0:
            # 阈值不完全匹配时取最近
            tt = test_df[test_df['run'] == run]
            idx = (tt['threshold'] - thr).abs().idxmin()
            sub = tt.loc[[idx]]
        rows.append({'run': run, 'target': k, 'test_dice': sub[f'dice_v{k}'].mean(), 'thr': thr})
    res = pd.DataFrame(rows)
    # 按 arm 汇总
    res['arm'] = res['run'].str.replace('_fold\\d+$', '', regex=True)
    return res

for dname, vdir, tdir, n, arms in [
    ('QUBIQ-kidney', 'outputs/analysis/qubiq_kidney', 'outputs/analysis/qubiq_kidney_test', 3,
     ['qubiq_kidney_A', 'qubiq_kidney_B', 'qubiq_kidney_C']),
    ('QUBIQ-brain', 'outputs/analysis/qubiq_brain', 'outputs/analysis/qubiq_brain_test', 7,
     ['qubiq_brain_A', 'qubiq_brain_B', 'qubiq_brain_C']),
]:
    val = load(vdir, 'per_sample_val_fold*.csv')
    tst = load(tdir, 'per_sample_test.csv')
    cal = calibrate_thresholds(val, n)
    res = report_test(tst, cal, n)
    print(f"\n### {dname} 保留测试集（{tst['case_id'].nunique()} 例，验证折标定阈值）")
    print(f"  测试集行: {len(tst)}  case: {tst['case_id'].nunique()}  run: {tst['run'].nunique()}")
    # 每个 arm 的协议跨度（V>=1 vs V>=N）和 train-target 效应（@V>=N）
    print(f"  评测靶列: V>=1 ... V>={n}")
    for arm in arms:
        sub = res[res['arm'] == arm]
        if sub.empty:
            continue
        line = f"  {arm:16s}"
        vals = {}
        for k in range(1, n + 1):
            v = sub[sub['target'] == k]['test_dice'].mean()
            vals[k] = v
            line += f"  V{k}={v:.3f}"
        print(line)
        print(f"      -> 协议跨度 V1-V{n} = {(vals[1]-vals[n])*100:.2f} 点")
    # 训练靶效应 @V>=n
    vn = {}
    for arm in arms:
        sub = res[res['arm'] == arm]
        vn[arm] = sub[sub['target'] == n]['test_dice'].mean()
    print(f"  训练靶效应 @V>={n} = {(max(vn.values())-min(vn.values()))*100:.2f} 点")
    # 保存
    res.to_csv(os.path.join(tdir, 'test_reported.csv'), index=False)
    print(f"  -> 已存 {tdir}/test_reported.csv")
