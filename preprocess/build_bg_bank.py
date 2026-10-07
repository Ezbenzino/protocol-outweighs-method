"""预提取无结节背景 patch 到磁盘 bank（一次性），供 P1-3 bg 训练快速加载。

背景：LIDC npz 为压缩存储，训练中逐个 slice 读取触发整组解压（40x 慢）。
本脚本将无结节切片上的 128x128 patch 提取到 data/processed/bg_bank.npz。
"""
import numpy as np
import os, glob, time

data_dir = "data/processed/npz"
out_path = "data/processed/bg_bank.npz"
ph = 128
n_target = 2500
per_case = 3

t0 = time.time()
images, votes = [], []
files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
print(f"[build_bg_bank] 扫描 {len(files)} 个 case", flush=True)

for fi, f in enumerate(files):
    if len(images) >= n_target:
        break
    z = np.load(f, mmap_mode="r")
    try:
        n_slices = int(z["slices"].shape[0])
        h, w = int(z["slices"].shape[1]), int(z["slices"].shape[2])
        if h < ph or w < ph:
            continue
        consensus = np.asarray(z["consensus"])  # 解压一次
        if consensus.ndim != 3:
            continue
        flat = consensus.reshape(n_slices, -1)
        zero_slices = np.where(flat.max(axis=1) == 0)[0]
        if len(zero_slices) == 0:
            continue
        np.random.shuffle(zero_slices)
        slices_all = np.asarray(z["slices"])  # 解压一次
        added = 0
        for sl in zero_slices[: per_case * 3]:
            img_s = slices_all[sl]
            vote_s = consensus[sl]
            cy = np.random.randint(ph // 2, h - ph // 2)
            cx = np.random.randint(ph // 2, w - ph // 2)
            pi = img_s[cy - ph // 2: cy + ph // 2, cx - ph // 2: cx + ph // 2].copy()
            pv = vote_s[cy - ph // 2: cy + ph // 2, cx - ph // 2: cx + ph // 2].copy()
            images.append(pi)
            votes.append(pv)
            added += 1
            if added >= per_case:
                break
    except Exception as e:
        print(f"  skip {os.path.basename(f)}: {e}", flush=True)
        continue
    if (fi + 1) % 100 == 0:
        print(f"  {fi+1}/{len(files)} case, 已提取 {len(images)} patch, {time.time()-t0:.0f}s", flush=True)

print(f"[build_bg_bank] 共提取 {len(images)} 个背景 patch，保存中...", flush=True)
np.savez_compressed(out_path, images=np.stack(images), votes=np.stack(votes))
print(f"[build_bg_bank] 已保存 {out_path}，总耗时 {time.time()-t0:.0f}s", flush=True)
