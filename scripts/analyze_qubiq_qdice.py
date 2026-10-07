# QUBIQ 官方 Q-Dice 汇总（论文附录用）
# 输入: outputs/analysis/qubiq_{kidney,brain}_qdice/qdice_val_fold{0-4}.csv / qdice_test.csv
# 输出: 训练臂 x 数据集的 Q-Dice（5 折 val 均值±std；保留 test 均值）
# 口径: Q-Dice = mean over thr in {0.1..0.9} of Dice(连续GT=V/N >= thr, pred >= thr)
import glob, os
import numpy as np
import pandas as pd

DATASETS = {
    "QUBIQ-kidney (CT, N=3)": dict(dir="outputs/analysis/qubiq_kidney_qdice",
                                   arms=["qubiq_kidney_A", "qubiq_kidney_B", "qubiq_kidney_C"]),
    "QUBIQ-brain (MRI, N=7)": dict(dir="outputs/analysis/qubiq_brain_qdice",
                                   arms=["qubiq_brain_A", "qubiq_brain_B", "qubiq_brain_C"]),
}

def load_rows(cfg, split_tag):
    rows = []
    pat = os.path.join(cfg["dir"], f"qdice_{split_tag}*.csv")
    for fp in sorted(glob.glob(pat)):
        rows.append(pd.read_csv(fp))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)

print("=" * 70)
print("QUBIQ 官方 Q-Dice（论文附录）—— 连续 GT=多标注者平均(V/N)，阈值 0.1~0.9 平均")
print("定义: arXiv:2405.18435 / qubiq.grand-challenge.org/participation")
print("=" * 70)
for dname, cfg in DATASETS.items():
    val = load_rows(cfg, "val_fold")
    test = load_rows(cfg, "test")
    print(f"\n### {dname}")
    print(f"{'训练臂'.ljust(22)}{'5折val Q-Dice'.rjust(18)}{'保留test Q-Dice'.rjust(18)}")
    for arm in cfg["arms"]:
        vv = val[val["run"].str.startswith(arm)]["qdice"]
        tt = test[test["run"].str.startswith(arm)]["qdice"]
        vs = f"{vv.mean():.4f} ± {vv.std():.4f} (n={len(vv)})" if len(vv) else "—"
        ts = f"{tt.mean():.4f} ± {tt.std():.4f} (n={len(tt)})" if len(tt) else "—"
        print(f"{arm.ljust(22)}{vs.rjust(18)}{ts.rjust(18)}")
    # 训练靶效应（Q-Dice 口径）
    va = val[val["run"].str.startswith(cfg["arms"][0])]["qdice"].mean()
    vc = val[val["run"].str.startswith(cfg["arms"][-1])]["qdice"].mean()
    ta = test[test["run"].str.startswith(cfg["arms"][0])]["qdice"].mean()
    tc = test[test["run"].str.startswith(cfg["arms"][-1])]["qdice"].mean()
    print(f"  训练靶效应 C−A: val {(vc-va)*100:.2f} 点 | test {(tc-ta)*100:.2f} 点")
print("\n" + "=" * 70)
