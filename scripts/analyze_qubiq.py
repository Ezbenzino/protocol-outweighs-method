# QUBIQ 外部验证分析：对称评测矩阵 -> 协议效应分解
# 输入: outputs/analysis/qubiq_{kidney,brain}/per_sample_val_fold{0-4}.csv
# 输出: 训练靶(行) x 评测靶(列) 的 Dice@最优阈值 / Dice@0.5；并量化协议效应 vs 训练靶效应
import glob, os
import numpy as np
import pandas as pd

DATASETS = {
    "QUBIQ-kidney (CT, N=3)": dict(dir="outputs/analysis/qubiq_kidney", n=3, arms=["A_union", "B_majority", "C_consensus"], prefix="qubiq_kidney"),
    "QUBIQ-brain (MRI, N=7)": dict(dir="outputs/analysis/qubiq_brain", n=7, arms=["A_union", "B_majority", "C_consensus"], prefix="qubiq_brain"),
}
# LIDC 主结论（2026-08-31 复核修正：训练靶效应为 2.99 而非 5.34，架构效应为 1.64 而非 2.99；
# 原 2.99 被错位到架构格、5.34 为无来源数字。来源：outputs/analysis/symmetric_analysis.json 重算：
# tgtA–tgtE @V>=4 最优阈值 0.8025-0.7726=2.99；tgtB/R18/PU/CNX/PVT @V>=1 最优 0.8535-0.8371=1.64）
LIDC_REF = {"协议(评测靶)效应": 16.42, "训练靶效应": 2.99, "架构效应": 1.64}

def load_table(cfg):
    """返回 (case_count, df): 每个 (run, threshold) 聚合样本平均的 dice_vK"""
    rows = []
    for fp in sorted(glob.glob(os.path.join(cfg["dir"], "per_sample_val_fold*.csv"))):
        df = pd.read_csv(fp)
        rows.append(df)
    df = pd.concat(rows, ignore_index=True)
    # 按 (run, threshold) 聚合样本平均
    cols = [f"dice_v{k}" for k in range(1, cfg["n"] + 1)]
    # 分析单元 = 病例：先在病例内对实例平均，再跨病例平均（与 LIDC 侧同口径）
    agg = (df.groupby(["run", "threshold", "case_id"])[cols].mean()
             .groupby(["run", "threshold"]).mean().reset_index())
    return agg

def arm_summary(agg, run_prefix, n, thr_mode):
    """对某训练臂：每个评测靶 -> 5折最优阈值(或0.5)的平均Dice。返回 {target: dice}"""
    sub = agg[agg["run"].str.startswith(run_prefix)]
    out = {}
    for k in range(1, n + 1):
        col = f"dice_v{k}"
        if thr_mode == "best":
            best = sub.loc[sub.groupby("run")[col].idxmax(), col].mean()
        else:
            best = sub.loc[sub["threshold"] == 0.5, col].mean()
        out[k] = float(best)
    return out

print("=" * 70)
for dname, cfg in DATASETS.items():
    agg = load_table(cfg)
    n = cfg["n"]
    results = {}
    for arm in cfg["arms"]:
        results[arm] = arm_summary(agg, f"{cfg['prefix']}_{arm.split('_')[0]}_", n, "best")
    results_05 = {}
    for arm in cfg["arms"]:
        results_05[arm] = arm_summary(agg, f"{cfg['prefix']}_{arm.split('_')[0]}_", n, "05")

    print(f"\n### {dname}")
    print(f"评测靶列: V>=1 (并集) ... V>={n} (全体一致)")
    # 最优阈值表
    print(f"\n-- Dice@最优阈值（每个 run 各自选 33 个阈值中的最优）--")
    hdr = "训练靶".ljust(12) + "".join(f"V>={k}".rjust(9) for k in range(1, n + 1))
    print(hdr)
    for arm in cfg["arms"]:
        vals = results[arm]
        print(arm.ljust(12) + "".join(f"{vals[k]:.3f}".rjust(9) for k in range(1, n + 1)))
    # 0.5 表
    print(f"\n-- Dice@固定阈值0.5 --")
    print(hdr)
    for arm in cfg["arms"]:
        vals = results_05[arm]
        print(arm.ljust(12) + "".join(f"{vals[k]:.3f}".rjust(9) for k in range(1, n + 1)))

    # 效应分解（用最优阈值表）
    # 评测靶(协议)效应：同一训练靶下，V>=1 与 V>=N 的跨度；取3臂最大
    protocol_spans = []
    for arm in cfg["arms"]:
        v = results[arm]
        protocol_spans.append(abs(v[1] - v[n]))
    protocol_eff = max(protocol_spans) * 100
    # 训练靶效应：同一评测靶下，union vs majority/consensus 的跨度；取所有靶最大
    target_spans = []
    for k in range(1, n + 1):
        vs = [results[arm][k] for arm in cfg["arms"]]
        target_spans.append(max(vs) - min(vs))
    target_eff = max(target_spans) * 100

    print(f"\n-- 效应分解（最优阈值口径，单位: Dice 百分点）--")
    print(f"  评测靶(协议)效应(最大跨度) : {protocol_eff:.2f}   (LIDC 参考: {LIDC_REF['协议(评测靶)效应']})")
    print(f"  训练靶效应(最大跨度)      : {target_eff:.2f}   (LIDC 参考: {LIDC_REF['训练靶效应']})")
    print(f"  结论: 协议效应{'显著大于' if protocol_eff > target_eff * 2 else '约等于'}训练靶效应 -> "
          f"{'QUBIQ 复现 LIDC 主结论' if protocol_eff > target_eff else '与 LIDC 不同'}")
print("\n" + "=" * 70)
