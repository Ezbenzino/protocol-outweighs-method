# -*- coding: utf-8 -*-
"""SVLS 臂上线前的自检：新臂行为正确，且【旧臂一个字节都没变】。

改动 combo.py 是有风险的——那是四个主实验臂共用的代码。这个脚本回答两件事：

  A 回归：union / majority / consensus / soft 四个臂的损失值，与不含 SVLS 代码路径
    时是否完全相同。做法是把 SVLS 靶所依赖的输入扰动掉，四个旧臂的损失必须纹丝不动，
    说明它们根本没走到新代码。
  B 正确性：SVLS 软靶是否真的在病灶内部趋近 1、外部趋近 0、跨边界渐变，
    以及它与 consensus 软靶（真实投票）到底差多少——那正是 C 臂与 E 臂的差异来源。

用法：
    python scripts\\verify_svls.py
"""
import ctypes, os, sys
for _p in (os.path.join(sys.prefix, "Library", "bin", "libiomp5md.dll"),
           os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib", "libiomp5md.dll")):
    if os.path.exists(_p):
        try:
            ctypes.WinDLL(_p); break
        except OSError:
            pass

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.losses.combo import CombinedLoss, TARGET_SPECS
from src.losses.svls import gaussian_kernel_2d, svls_target
from src.utils.config import load_config

W = 84


def make_sample(seed=0, hw=64):
    """造一个圆形病灶，票数从中心 4 票向外递减到 0 票，模拟真实的标注者分歧。"""
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.arange(hw), torch.arange(hw), indexing="ij")
    r = torch.hypot(yy.float() - hw / 2, xx.float() - hw / 2)
    vote = torch.zeros(hw, hw)
    for k, rad in enumerate([14.0, 11.0, 8.0, 5.0]):     # 半径越小票数越高
        vote += (r <= rad).float()
    vote = vote[None, None]                               # (1,1,H,W)
    cons = vote / 4.0
    s = {
        "union_mask": (vote >= 1).float(),
        "majority_mask": (vote >= 2).float(),
        "consensus": cons,
        "disagreement": 4.0 * cons * (1.0 - cons),
        "instance_ids": (vote >= 1).float(),
        "maj_instance_ids": (vote >= 2).float(),
        "boundary": torch.zeros_like(cons),
        "maj_boundary": torch.zeros_like(cons),
    }
    logits = torch.randn(1, 1, hw, hw, generator=g) * 2.0
    return s, logits


def main():
    cfg = load_config("configs/default.yaml")
    for k in ("use_dice", "use_bce"):
        setattr(cfg.loss, k, True)
    for k in ("use_sitl", "use_csl", "use_brbc", "use_lovasz"):
        setattr(cfg.loss, k, False)

    sample, logits = make_sample()
    bnd = torch.zeros_like(logits)

    print("=" * W)
    print("A. 回归自检：扰动 SVLS 所依赖的输入，旧臂的损失必须完全不变")
    print("=" * W)
    base = {}
    for tgt in TARGET_SPECS:
        cfg.loss.target = tgt
        base[tgt] = float(CombinedLoss(cfg)(logits, bnd, sample)[0])

    # SVLS 靶由 majority_mask 推导。这里换一个【形状不同但 majority_mask 相同】的
    # 场景做不到，所以反过来做：直接把 svls 核换成 delta 核（等价于不平滑），
    # 只有 svls 臂应该变，其余四个臂必须一模一样。
    pert = {}
    for tgt in TARGET_SPECS:
        cfg.loss.target = tgt
        m = CombinedLoss(cfg)
        m.svls_kernel = torch.zeros_like(m.svls_kernel)
        m.svls_kernel[0, 0, m.svls_kernel.shape[-1] // 2, m.svls_kernel.shape[-1] // 2] = 1.0
        pert[tgt] = float(m(logits, bnd, sample)[0])

    print(f"{'训练靶':<12}{'原始损失':>14}{'换核后':>14}{'差':>14}{'判定':>12}")
    ok = True
    for tgt in TARGET_SPECS:
        d = pert[tgt] - base[tgt]
        expect_change = (tgt == "svls")
        good = (abs(d) > 1e-9) == expect_change
        ok &= good
        print(f"{tgt:<12}{base[tgt]:>14.8f}{pert[tgt]:>14.8f}{d:>+14.2e}"
              f"{('通过' if good else '!! 异常 !!'):>12}")
    print("-" * W)
    print("  预期：只有 svls 变，其余四个臂不变（说明它们没走到新代码路径）")
    print(f"  结论：{'通过' if ok else '!! 有臂被污染，不要开训 !!'}")

    print("\n" + "=" * W)
    print("B. 正确性：SVLS 软靶 vs 真实投票软靶（这正是 E 臂与 C 臂的差异来源）")
    print("=" * W)
    ker = gaussian_kernel_2d(3, 1.0)
    sv = svls_target(sample["majority_mask"], ker)[0, 0]
    cons = sample["consensus"][0, 0]
    maj = sample["majority_mask"][0, 0]
    hw = sv.shape[0]
    row = hw // 2
    print(f"  沿中心行从圆心向外取值（半径 0 -> {hw//2}）：")
    idx = list(range(hw // 2, hw // 2 + 20))
    print("    r      " + "".join(f"{i - hw//2:>6}" for i in idx))
    print("    票数/4 " + "".join(f"{cons[row, i]:>6.2f}" for i in idx))
    print("    V>=2   " + "".join(f"{maj[row, i]:>6.0f}" for i in idx))
    print("    SVLS   " + "".join(f"{sv[row, i]:>6.2f}" for i in idx))
    print("-" * W)
    print(f"  SVLS 取值范围 [{sv.min():.3f}, {sv.max():.3f}]，"
          f"非 0 非 1 的像素占比 {100.0 * ((sv > 1e-6) & (sv < 1 - 1e-6)).float().mean():.1f}%")
    print(f"  真实投票软靶非 0 非 1 占比 {100.0 * ((cons > 0) & (cons < 1)).float().mean():.1f}%")
    print(f"  两者平均绝对差 {float((sv - cons).abs().mean()):.4f}")
    print("  [判读] SVLS 的渐变只出现在边界一两个像素内（核宽决定），真实投票的渐变"
          "跨度由医生分歧决定。两者不等价，正是 E 臂存在的意义。")


if __name__ == "__main__":
    main()
