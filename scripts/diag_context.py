# -*- coding: utf-8 -*-
"""诊断：为什么"同一块 128 区域"在整片推理下的 Dice 会塌掉。

视野阶梯（analyze_scope.py 表 2）的结果是反直觉的：
    patch128 -> win128   -34 ~ -53 点   <== 评测范围一个像素没变，只换了模型输入
    win128   -> win512   -1.5 ~ -3.2 点 <== 评测范围放大 16 倍，几乎不要钱
所以塌陷的根源不是"评测视野"，是"模型输入"。本脚本把这一步再拆开。

对同一批结节，用四种输入喂给【同一个模型】，然后【都只在中心 128 区域】上算 Dice
（评测区域、GT、阈值全部固定，唯一变量是模型看到了什么）：

    A patch128    输入 = 128 裁剪                     —— 现有协议
    B canvas512   输入 = 512 全零画布，把那块 128 贴回原位置
                  —— 输入尺寸变成 512，但周围没有任何真实解剖内容
    C crop256     输入 = 256 裁剪                     —— 少量真实上下文
    D full512     输入 = 完整 512 切片                —— 整片推理

判读：
    B ~ A 且 D << A  ->  罪魁是【周围的真实解剖内容】。模型没学会把结节和肺内
                         其他结构分开，因为训练时它从未在同一张图里见过结节以外
                         的区域。这是训练数据构造的问题，不是评测口径的问题。
    B ~ D 且 D << A  ->  罪魁是【输入尺寸本身】。尺寸改变了网络的有效感受野与
                         特征尺度，属于 train-test 分辨率不匹配，与解剖内容无关。
    C 的位置说明退化是随上下文量渐变还是突变。

【关于 OMP Error #15】
这个 conda 环境里磁盘上有两份 libiomp5md.dll（MKL 一份、torch 一份），
见文件开头 _preload_single_openmp() 的注释。处理方式是在 import numpy / torch
之前先按全路径把其中一份装进进程，让后来者按同名复用它，进程里真的只剩一份。
【没有】使用 KMP_DUPLICATE_LIB_OK —— 那个变量是"允许两份共存别报错"，
Intel 自己说它可能静默产生错误结果，在需要精确数字的场合不能用。

用法（PowerShell，在仓库根目录下，约 1 分钟）：
    python scripts\\diag_context.py
"""
import ctypes
import os
import sys


def _preload_single_openmp():
    """在 import numpy / torch 之前，先把【唯一一份】OpenMP 运行时加载进进程。

    背景：这个 conda 环境里磁盘上有两份 libiomp5md.dll ——
        <env>\Library\bin\libiomp5md.dll                     (2.05 MB, MKL 用, numpy 依赖)
        <env>\Lib\site-packages\torch\lib\libiomp5md.dll    (1.61 MB, torch 自带)
    两边各自把自己那份加载进来，就触发 OMP: Error #15。

    这里做的事：用 ctypes 按【全路径】先把 MKL 那份装进进程。Windows 的模块表以
    DLL 的基名（libiomp5md.dll）为键，之后 torch 再去加载它自己那份时，加载器发现
    同名模块已在进程内，直接复用，不会产生第二份。

    这【不是】KMP_DUPLICATE_LIB_OK。那个变量的含义是"允许两份共存、别报错"，
    Intel 明确说它可能静默算错；这里是让进程里真的只剩一份运行时，
    正是报错信息自己建议的解法（"ensure that only a single OpenMP runtime is
    linked into the process"）。选 MKL 那份是因为它版本更新（2.05 MB vs 1.61 MB），
    且 OpenMP 运行时向后兼容。

    正确性不靠"应该没问题"来保证：跑完后 A_patch128 这一列会与
    per_sample_test.csv 逐条对账，数字对不上就说明这条路不能用。
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
W = 100


def roi_center_crop(arr, center, size):
    """以 (cy, cx) 为中心裁 size x size，越界处补零。

    与 src/data/utils.py 的同名函数逐行等价，此处自带一份以避免导入 scipy/skimage。
    """
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


def centre_crop(a, size):
    c = a.shape[0] // 2
    h = size // 2
    return a[c - h:c + h, c - h:c + h]


@torch.no_grad()
def run(model, dev, x, in_ch):
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None, None]
    if in_ch == 3:
        t = t.repeat(1, 3, 1, 1)
    return region_prob(model, t.to(dev)).float().cpu().numpy()[0, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--runs", default="tgtB_fold0,tgtA_fold0")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-nodules", type=int, default=40, help="抽多少个结节（够看趋势即可）")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--level", type=int, default=2, help="用 V>=level 作为 GT")
    ap.add_argument("--out", default="outputs/analysis_scope/diag_context.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_ch = cfg.model.in_channels
    with open(os.path.join(cfg.data.split_dir, f"{args.split}.csv"), encoding="utf-8") as f:
        cases = [ln.strip() for ln in f if ln.strip() and ln.strip() != "case_id"]

    # 每个病例取中间那个结节，避免样本全部来自同一个病例
    jobs = []
    for c in cases:
        mp = os.path.join(cfg.data.npz_dir, f"{c}.json")
        npz = os.path.join(cfg.data.npz_dir, f"{c}.npz")
        if not (os.path.exists(mp) and os.path.exists(npz)):
            continue
        with open(mp, encoding="utf-8") as f:
            nods = json.load(f).get("nodules", [])
        if nods:
            jobs.append((c, nods[len(nods) // 2]))
        if len(jobs) >= args.n_nodules:
            break
    print(f"OpenMP 运行时: {_OMP_DLL or '未预载（非 Windows 或未找到）'}")
    print(f"device={dev}  结节 {len(jobs)} 个（每例取中间那个）  "
          f"GT=V>={args.level}  阈值={args.thr}", flush=True)

    KEYS = ("A_patch128", "B_canvas512", "C_crop256", "D_full512")
    rep = {"n_requested": len(jobs), "thr": args.thr, "level": args.level, "runs": {}}
    for run_name in [r.strip() for r in args.runs.split(",") if r.strip()]:
        ck = os.path.join(cfg.outputs.root, "runs", run_name, "checkpoints", "best.pth")
        if not os.path.exists(ck):
            print(f"  [跳过] 找不到 {ck}")
            continue
        model, arch, act = load_for_eval(ck, cfg)
        model.to(dev).eval()

        recs = []
        for c, nod in jobs:
            z = np.load(os.path.join(cfg.data.npz_dir, f"{c}.npz"), mmap_mode="r")
            si = int(nod["slice"])
            cy, cx = float(nod["cy"]), float(nod["cx"])
            img = z["slices"][si].astype(np.float32) / 255.0
            vote = z["consensus"][si]
            gt = roi_center_crop(vote, (cy, cx), PATCH) >= args.level
            if gt.sum() == 0:
                continue
            iy, ix = int(round(cy)), int(round(cx))
            H, Wd = img.shape
            if not (64 <= iy <= H - 64 and 64 <= ix <= Wd - 64):
                continue                      # 贴边结节，窗口取法不一致，跳过

            pa = run(model, dev, roi_center_crop(img, (cy, cx), PATCH), in_ch)

            canvas = np.zeros_like(img)
            canvas[iy - 64:iy + 64, ix - 64:ix + 64] = img[iy - 64:iy + 64, ix - 64:ix + 64]
            pb = run(model, dev, canvas, in_ch)[iy - 64:iy + 64, ix - 64:ix + 64]

            pc = centre_crop(run(model, dev, roi_center_crop(img, (cy, cx), 256), in_ch), PATCH)
            pd = run(model, dev, img, in_ch)[iy - 64:iy + 64, ix - 64:ix + 64]

            r = {"case_id": c, "slice": si, "diameter_mm": float(nod["diameter_mm"]),
                 "gt_area": int(gt.sum())}
            for k, p in zip(KEYS, (pa, pb, pc, pd)):
                r[k] = dice_at(p, gt, args.thr)
                r[k + "_meanprob"] = float(p.mean())
                if k != "A_patch128":
                    r[k + "_corrA"] = float(np.corrcoef(p.ravel(), pa.ravel())[0, 1])
            recs.append(r)

        print("\n" + "=" * W)
        print(f"[{run_name}] arch={arch}   n={len(recs)}   "
              f"评测区域固定为中心 128，唯一变量是模型输入")
        print("=" * W)
        print(f"{'输入':<16}{'Dice':>9}{'相对 A':>10}{'平均预测概率':>15}{'与 A 的相关':>14}")
        base = float(np.mean([r["A_patch128"] for r in recs])) if recs else float("nan")
        summ = {}
        for k in KEYS:
            d = float(np.mean([r[k] for r in recs]))
            mp_ = float(np.mean([r[k + "_meanprob"] for r in recs]))
            cc = (float(np.mean([r[k + "_corrA"] for r in recs]))
                  if k != "A_patch128" else None)
            print(f"{k:<16}{d:>9.4f}{(d - base) * 100:>+10.2f}{mp_:>15.4f}"
                  f"{(f'{cc:.3f}' if cc is not None else '  -  '):>14}")
            summ[k] = dict(dice=d, delta_pts=(d - base) * 100, mean_prob=mp_, corr_with_A=cc)

        dA, dB, dC, dD = (summ[k]["dice"] for k in KEYS)
        print("-" * W)
        print(f"  A->B {100*(dB-dA):+.2f}   B->C {100*(dC-dB):+.2f}   "
              f"C->D {100*(dD-dC):+.2f}   合计 A->D {100*(dD-dA):+.2f} 点")
        if dB > dA - 0.05:
            v = "周围真实解剖内容"
            print("  判读: B(空画布)基本保住了 A -> 罪魁是【周围的真实解剖内容】，不是输入尺寸。")
            print("        模型没学会把结节和肺内其他结构分开：训练时它从未在同一张图里")
            print("        见过结节以外的区域。这是训练数据构造的问题，不是评测口径的问题。")
        elif abs(dB - dD) < 0.05:
            v = "输入尺寸本身"
            print("  判读: B(空画布)和 D(整片)一样差 -> 罪魁是【输入尺寸本身】，")
            print("        属于 train-test 分辨率/感受野不匹配，与解剖内容无关。")
        else:
            v = "两者兼有"
            print("  判读: 两个因素都有贡献，按上表差值分摊（A->B 是尺寸项，B->D 是内容项）。")
        rep["runs"][run_name] = dict(arch=arch, n=len(recs), verdict=v,
                                     summary=summ, per_nodule=recs)
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    print(f"\n已保存 -> {args.out}")
    print("[核对] per_nodule 里存了每个结节的 case_id 与 diameter_mm，"
          "可与 per_sample_test.csv 逐条对账，确认 A 这一列复现了现有协议。")


if __name__ == "__main__":
    main()
