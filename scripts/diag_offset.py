# -*- coding: utf-8 -*-
"""诊断：模型是否学到了"结节一定在图正中"这个位置先验。

【为什么必须先做这个】
diag_context.py 已经证明：把同一个模型的输入从 128 裁剪换成 512 整片，
同一块 128 区域内的 Dice 掉 26~44 点，而且主因是【输入尺寸】而非周围解剖内容
（512 全零画布就已经掉了 25.7/36.7 点），塌陷方向是【漏检】而非假阳性泛滥。

由此想到的补救方案是滑窗推理：把整片切成 128 的窗口逐块推理再拼接，
让模型始终工作在训练时的输入尺寸下。但这个方案有个前提没验证过——
训练用的 patch 全部是【以结节为中心】裁出来的，模型完全可能顺带学到
"目标就在图正中"。若真如此，滑窗里那些结节偏离中心的窗口照样失效，
滑窗方案不成立，用这批模型根本测不了评测空间因素。

【测法】
把 128 评测窗口的中心从结节中心平移 d 个像素（GT 跟着一起平移，
所以窗口里的内容和标注完全对应，唯一变化的是结节在图中的位置），
看 Dice 随 d 怎么变。四个方向（上下左右）+ 两条对角各测一遍再平均，
避免某个方向恰好撞上其他解剖结构造成的偶然。

判读：
    Dice 基本持平        -> 无位置先验，滑窗方案可行，可以去做 F3 的干净版本
    Dice 随 d 明显下降   -> 有位置先验。滑窗窗口里偏离中心的结节会被漏掉，
                            这批模型测不了整片级的评测空间因素，4.10 节要改写

d=0 这一档必须精确复现现有协议的逐结节 Dice——这是本脚本的自检。

用法（PowerShell，在仓库根目录下，约 2 分钟）：
    python scripts\\diag_offset.py
"""
import ctypes
import os
import sys


def _preload_single_openmp():
    """在 import numpy / torch 之前，把【唯一一份】OpenMP 运行时加载进进程。

    这个 conda 环境磁盘上有两份 libiomp5md.dll（MKL 一份、torch 一份），
    两边各加载各的就触发 OMP: Error #15。Windows 的模块表以 DLL 基名为键，
    先按全路径装好 MKL 那份，torch 之后再加载同名 DLL 时会直接复用，
    进程里真的只剩一份运行时。

    这【不是】KMP_DUPLICATE_LIB_OK（那是"允许两份共存别报错"，Intel 说它
    可能静默算错）。本方案的数值等价性已经实测过：diag_context.py 的
    A_patch128 与 per_sample_test.csv 逐条对账，40/40 条最大绝对差 2.9e-08，
    属 float32 舍入量级。
    """
    if sys.platform != "win32":
        return None
    for p in (os.path.join(sys.prefix, "Library", "bin", "libiomp5md.dll"),
              os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib",
                           "libiomp5md.dll")):
        if os.path.exists(p):
            try:
                ctypes.WinDLL(p)
                return p
            except OSError:
                continue
    return None


_OMP_DLL = _preload_single_openmp()

import argparse
import json

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.from_ckpt import load_for_eval, region_prob
from src.utils.config import load_config

PATCH = 128
# 平移量（像素）。128 窗口、半径 64，结节中位等效直径约 7 mm（约 10 px），
# 平移到 56 时结节距窗口边缘还有约 3 px，仍完整在窗内。
OFFSETS = (0, 8, 16, 24, 32, 40, 48, 56)
# 六个方向：上下左右 + 两条对角。对角按 1/sqrt(2) 分解，保证位移模长与 d 一致。
_S = 0.70710678
DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1), (_S, _S), (-_S, -_S))
W = 100


def roi_center_crop(arr, center, size):
    """以 (cy, cx) 为中心裁 size x size，越界处补零；与 src/data/utils.py 逐行等价。"""
    cy, cx = int(round(center[0])), int(round(center[1]))
    arr = np.asarray(arr)
    H, Wd = arr.shape
    y0, x0 = cy - size // 2, cx - size // 2
    y1, x1 = y0 + size, x0 + size
    pt, pb = max(0, -y0), max(0, y1 - H)
    pl, pr = max(0, -x0), max(0, x1 - Wd)
    crop = arr[max(0, y0):min(H, y1), max(0, x0):min(Wd, x1)]
    if pt or pb or pl or pr:
        crop = np.pad(crop, ((pt, pb), (pl, pr)))
    return crop


def dice_at(prob, gt, thr):
    b = prob > thr
    s = b.sum() + gt.sum()
    return float(2.0 * np.logical_and(b, gt).sum() / s) if s > 0 else 1.0


@torch.no_grad()
def run_batch(model, dev, xs, in_ch):
    t = torch.from_numpy(np.stack(xs).astype(np.float32))[:, None]
    if in_ch == 3:
        t = t.repeat(1, 3, 1, 1)
    return region_prob(model, t.to(dev)).float().cpu().numpy()[:, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--runs", default="tgtB_fold0,tgtA_fold0")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-nodules", type=int, default=120)
    ap.add_argument("--thr", type=float, default=None,
                    help="固定阈值。默认不固定：每个臂用它【自己】的验证集校准阈值")
    ap.add_argument("--thr-from", default="outputs/analysis_full/symmetric_analysis.json",  # 2026-10-02: 旧默认 outputs/analysis/ 是 08-31 的过期副本（实例级校准阈值）
                    help="各臂各层级的校准阈值来源。两个臂的最优阈值不同"
                         "（tgtB@V>=2 是 0.50，tgtBshift 是 0.45），"
                         "对双方都用 0.5 会系统性压低平移臂，这个对照必须各用各的")
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--out", default="outputs/analysis_scope/diag_offset.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_ch = cfg.model.in_channels
    with open(os.path.join(cfg.data.split_dir, f"{args.split}.csv"), encoding="utf-8") as f:
        cases = [ln.strip() for ln in f if ln.strip() and ln.strip() != "case_id"]

    # 每个病例取中间那个结节；不够 n 个就再取次中间的，尽量摊到多个病例上
    jobs = []
    for pick in (0, 1, 2):
        for c in cases:
            mp = os.path.join(cfg.data.npz_dir, f"{c}.json")
            npz = os.path.join(cfg.data.npz_dir, f"{c}.npz")
            if not (os.path.exists(mp) and os.path.exists(npz)):
                continue
            with open(mp, encoding="utf-8") as f:
                nods = json.load(f).get("nodules", [])
            i = len(nods) // 2 + pick
            if 0 <= i < len(nods):
                jobs.append((c, nods[i]))
            if len(jobs) >= args.n_nodules:
                break
        if len(jobs) >= args.n_nodules:
            break

    print(f"OpenMP 运行时: {_OMP_DLL or '未预载'}")
    print(f"device={dev}  结节 {len(jobs)} 个  GT=V>={args.level}  "
          f"阈值={'固定 %g' % args.thr if args.thr is not None else '各臂用自己的校准值'}")
    print(f"平移量 {OFFSETS} px，每档 {len(DIRS)} 个方向取平均", flush=True)

    thr_src = None
    if args.thr is None:
        with open(args.thr_from, encoding="utf-8") as f:
            thr_src = json.load(f)["table1"]

    def thr_for(run_name):
        if args.thr is not None:
            return float(args.thr), "固定"
        arm = run_name.split("_fold")[0]
        if arm not in thr_src:
            raise SystemExit(f"{arm} 在 {args.thr_from} 里没有校准阈值；"
                             f"先跑 eval_symmetric.py + analyze_symmetric.py")
        return float(thr_src[arm][str(args.level)]["thr"]), "校准"

    rep = {"offsets": list(OFFSETS), "n_dirs": len(DIRS),
           "thr_mode": "fixed" if args.thr is not None else "per-arm-calibrated",
           "level": args.level, "runs": {}}
    for run_name in [r.strip() for r in args.runs.split(",") if r.strip()]:
        ck = os.path.join(cfg.outputs.root, "runs", run_name, "checkpoints", "best.pth")
        if not os.path.exists(ck):
            print(f"  [跳过] 找不到 {ck}")
            continue
        model, arch, act = load_for_eval(ck, cfg)
        model.to(dev).eval()
        thr_run, thr_kind = thr_for(run_name)

        recs = []
        for c, nod in jobs:
            z = np.load(os.path.join(cfg.data.npz_dir, f"{c}.npz"), mmap_mode="r")
            si = int(nod["slice"])
            cy, cx = float(nod["cy"]), float(nod["cx"])
            img = z["slices"][si].astype(np.float32) / 255.0
            vote = z["consensus"][si]
            H, Wd = img.shape
            # 最大平移后窗口仍需完整落在图内，否则补零面积会随 d 变化，混进新的变量
            m = 64 + max(OFFSETS)
            if not (m <= cy <= H - m and m <= cx <= Wd - m):
                continue
            if (roi_center_crop(vote, (cy, cx), PATCH) >= args.level).sum() == 0:
                continue

            xs, gts, keys = [], [], []
            for d in OFFSETS:
                for di, (uy, ux) in enumerate(DIRS):
                    if d == 0 and di > 0:
                        continue                      # d=0 时六个方向是同一个窗口
                    ctr = (cy + uy * d, cx + ux * d)
                    xs.append(roi_center_crop(img, ctr, PATCH))
                    gts.append(roi_center_crop(vote, ctr, PATCH) >= args.level)
                    keys.append((d, di))
            probs = []
            for a in range(0, len(xs), args.batch):
                probs.append(run_batch(model, dev, xs[a:a + args.batch], in_ch))
            probs = np.concatenate(probs, 0)

            r = {"case_id": c, "slice": si, "diameter_mm": float(nod["diameter_mm"])}
            per_d = {}
            for (d, di), p, g in zip(keys, probs, gts):
                per_d.setdefault(d, []).append(dice_at(p, g, thr_run))
            for d in OFFSETS:
                r[f"d{d}"] = float(np.mean(per_d[d]))
            r["d0_gt_area"] = int(gts[0].sum())
            recs.append(r)

        print("\n" + "=" * W)
        print(f"[{run_name}] arch={arch}   n={len(recs)}   阈值={thr_run:g}({thr_kind})   "
              f"窗口内容与 GT 完全对应，唯一变量是结节偏离窗口中心多少")
        print("=" * W)
        print(f"{'平移 d (px)':<14}" + "".join(f"{d:>10}" for d in OFFSETS))
        means = [float(np.mean([r[f"d{d}"] for r in recs])) for d in OFFSETS]
        print(f"{'Dice':<14}" + "".join(f"{m:>10.4f}" for m in means))
        print(f"{'相对 d=0':<14}" + "".join(f"{(m - means[0]) * 100:>+10.2f}" for m in means))
        rep["runs"][run_name] = {"arch": arch, "n": len(recs), "thr": thr_run,
                                 "mean_by_offset": dict(zip(map(str, OFFSETS), means)),
                                 "per_nodule": recs}
        print("-" * W)
        drop = (means[0] - means[-1]) * 100
        half = next((d for d, m in zip(OFFSETS, means) if m < means[0] * 0.9), None)
        print(f"  d=0 -> d={OFFSETS[-1]} 共下降 {drop:.2f} 点；"
              f"跌破 d=0 的 90% 发生在 d={half if half is not None else '未跌破'}")
        if drop < 3:
            print("  判读: 无明显位置先验 -> 滑窗推理方案可行，可以去做 F3 的干净版本。")
        elif drop > 10:
            print("  判读: 存在明显位置先验。模型依赖'目标在图正中'这个训练时的伪线索，")
            print("        滑窗里偏离中心的结节同样会被漏掉，这批模型测不了整片级的")
            print("        评测空间因素。4.10 节需要改写为'协议耦合'的负面结果。")
        else:
            print("  判读: 位置先验中等，滑窗可行但需在正文报告这项衰减。")
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    print(f"\n已保存 -> {args.out}")
    print("[自检] d=0 一列会与 per_sample_test.csv 逐条对账，确认复现现有协议。")


if __name__ == "__main__":
    main()
