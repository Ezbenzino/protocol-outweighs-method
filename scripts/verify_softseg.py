# -*- coding: utf-8 -*-
"""SoftSeg 上线前的自检。不通过就【不要训】，当场砍掉这一臂。

改动波及面很大——activation.py 新建、combo.py、trainer.py、evaluator.py、
四个评测脚本全部动过。这些都是五个已完成实验臂共用的代码路径。
所以本脚本的第一职责不是验证 SoftSeg 好不好，而是验证【旧臂没被碰坏】。

A 回归：五个旧臂（union/majority/consensus/soft/svls）的损失值必须与改动前
  逐位相同。它们的 final_activation 都是 sigmoid，走的应该是完全相同的代码路径。
B 活化函数：sigmoid 与归一化 ReLU 的取值范围、全零输入不产 NaN。
C Adaptive Wing：自身对自身为 0、随机对随机为正、对目标值可导没有 NaN。
D 拦截：非 sigmoid 活化下启用 BCE 必须【报错】而不是静默算出无意义的数。
E 向后兼容：不带 meta 的老 checkpoint 必须被判定为 sigmoid。

用法：
    python scripts\\verify_softseg.py
"""
import ctypes, os, sys
for _p in (os.path.join(sys.prefix, "Library", "bin", "libiomp5md.dll"),
           os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib", "libiomp5md.dll")):
    if os.path.exists(_p):
        try:
            ctypes.WinDLL(_p); break
        except OSError:
            pass

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.losses.adaptive_wing import adaptive_wing_loss
from src.losses.combo import CombinedLoss, TARGET_SPECS
from src.models.activation import activation_of, to_prob
from src.utils.config import load_config

W = 86
# 基准值：由 verify_svls.py 在同一个合成样本上算出的、【改 SoftSeg 之前】的事实。
# 对不上就是把旧臂改坏了，不是期望值。
BASELINE = {"union": 1.81630158, "majority": 1.91222405,
            "consensus": 1.90460730, "soft": 1.92209661}

# svls 单独处理。那个基准是用【普通高斯核】算的；之后核按原文改成了
# "中心权重抬升"版（src/losses/svls.py 的 center_boost），损失值必然变化。
# 所以对 svls 的正确检验不是"值没变"，而是：
#     关掉 center_boost 必须【精确复现】旧基准
# 这样才能证明差异只来自那一次有意的核修改，而不是别处漏改混进来的。
SVLS_BASELINE_PLAIN_KERNEL = 1.91324782


def make_sample(seed=0, hw=64):
    """与 verify_svls.py 完全相同的合成样本，保证基准值可比。"""
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.arange(hw), torch.arange(hw), indexing="ij")
    r = torch.hypot(yy.float() - hw / 2, xx.float() - hw / 2)
    vote = torch.zeros(hw, hw)
    for rad in (14.0, 11.0, 8.0, 5.0):
        vote += (r <= rad).float()
    vote = vote[None, None]
    cons = vote / 4.0
    s = {"union_mask": (vote >= 1).float(), "majority_mask": (vote >= 2).float(),
         "consensus": cons, "disagreement": 4.0 * cons * (1.0 - cons),
         "instance_ids": (vote >= 1).float(), "maj_instance_ids": (vote >= 2).float(),
         "boundary": torch.zeros_like(cons), "maj_boundary": torch.zeros_like(cons)}
    return s, torch.randn(1, 1, hw, hw, generator=g) * 2.0


def main():
    cfg = load_config("configs/default.yaml")
    cfg.loss.use_dice = cfg.loss.use_bce = True
    for k in ("use_sitl", "use_csl", "use_brbc", "use_lovasz"):
        setattr(cfg.loss, k, False)
    setattr(cfg.loss, "use_awing", False)
    sample, logits = make_sample()
    bnd = torch.zeros_like(logits)
    ok_all = True

    print("=" * W)
    print("A. 回归：五个旧臂的损失必须与改动前逐位相同")
    print("=" * W)
    print(f"{'训练靶':<12}{'改动前基准':>16}{'现在':>16}{'差':>14}{'判定':>10}")
    cfg.model.final_activation = "sigmoid"
    for tgt, base in BASELINE.items():
        cfg.loss.target = tgt
        v = float(CombinedLoss(cfg)(logits, bnd, sample)[0])
        d = v - base
        good = abs(d) < 1e-7
        ok_all &= good
        print(f"{tgt:<12}{base:>16.8f}{v:>16.8f}{d:>+14.2e}"
              f"{('通过' if good else '!! 改坏了 !!'):>10}")

    # svls：核已按原文有意修改过，检验方式是"退回旧核必须复现旧值"
    cfg.loss.target = "svls"
    if not hasattr(cfg.loss, "svls"):
        cfg.loss.svls = type("O", (), {})()
    cfg.loss.svls.ksize, cfg.loss.svls.sigma = 3, 1.0
    cfg.loss.svls.center_boost = False
    v_plain = float(CombinedLoss(cfg)(logits, bnd, sample)[0])
    cfg.loss.svls.center_boost = True
    v_boost = float(CombinedLoss(cfg)(logits, bnd, sample)[0])
    d = v_plain - SVLS_BASELINE_PLAIN_KERNEL
    good = abs(d) < 1e-7
    ok_all &= good
    print(f"{'svls(旧核)':<12}{SVLS_BASELINE_PLAIN_KERNEL:>16.8f}{v_plain:>16.8f}"
          f"{d:>+14.2e}{('通过' if good else '!! 改坏了 !!'):>10}")
    print(f"{'svls(原文核)':<12}{'—':>16}{v_boost:>16.8f}{v_boost - v_plain:>+14.2e}"
          f"{'有意改动':>10}")
    print("-" * W)
    print("  [说明] svls 的核已按原文改为中心权重抬升版，损失值本就该变。")
    print("         检验方式是关掉 center_boost 能否精确复现旧值 —— 能，")
    print("         就证明差异只来自那一次有意修改，没有别处漏改混进来。")

    print("\n" + "=" * W)
    print("B. 活化函数")
    print("=" * W)
    x = torch.randn(4, 1, 16, 16)
    ps, pr = to_prob(x, "sigmoid"), to_prob(x, "relu_norm")
    z = to_prob(torch.zeros(2, 1, 8, 8), "relu_norm")
    b_ok = (0 <= float(ps.min()) and float(ps.max()) <= 1
            and 0 <= float(pr.min()) and abs(float(pr.max()) - 1.0) < 1e-6
            and bool(torch.isfinite(z).all()) and float(z.abs().max()) == 0.0)
    ok_all &= b_ok
    print(f"  sigmoid   范围 [{float(ps.min()):.4f}, {float(ps.max()):.4f}]")
    print(f"  relu_norm 范围 [{float(pr.min()):.4f}, {float(pr.max()):.4f}]  "
          f"逐样本最大值 {[round(v, 4) for v in pr.flatten(1).max(1).values.tolist()]}")
    print(f"  全零输入 -> 全零图且无 NaN: {bool(torch.isfinite(z).all()) and float(z.abs().max()) == 0.0}")
    print(f"  判定: {'通过' if b_ok else '!! 异常 !!'}")

    print("\n" + "=" * W)
    print("C. Adaptive Wing 损失")
    print("=" * W)
    t = torch.rand(4, 1, 16, 16)
    self_l = float(adaptive_wing_loss(t, t))
    rand_l = float(adaptive_wing_loss(torch.rand_like(t), t))
    p = torch.rand(4, 1, 16, 16, requires_grad=True)
    adaptive_wing_loss(p, t).backward()
    c_ok = self_l < 1e-6 and rand_l > self_l and bool(torch.isfinite(p.grad).all())
    ok_all &= c_ok
    print(f"  自身对自身 {self_l:.3e}（应为 0）   随机对目标 {rand_l:.4f}（应为正）")
    print(f"  梯度有限: {bool(torch.isfinite(p.grad).all())}")
    print(f"  判定: {'通过' if c_ok else '!! 异常 !!'}")

    print("\n" + "=" * W)
    print("D. 拦截：非 sigmoid 活化下启用 BCE 必须报错，不能静默算错")
    print("=" * W)
    cfg.loss.target = "softseg"
    cfg.model.final_activation = "relu_norm"
    cfg.loss.use_bce = True
    try:
        CombinedLoss(cfg)
        d_ok = False
        print("  !! 没有拦住，会静默算出无意义的 BCE !!")
    except ValueError as e:
        d_ok = True
        print(f"  已拦截: {str(e)[:70]}")
    ok_all &= d_ok
    print(f"  判定: {'通过' if d_ok else '!! 异常 !!'}")

    print("\n" + "=" * W)
    print("E. SoftSeg 臂能否正常前向 + 向后兼容")
    print("=" * W)
    cfg.loss.use_bce = cfg.loss.use_dice = False
    cfg.loss.use_awing = True
    m = CombinedLoss(cfg)
    tot, parts = m(logits, bnd, sample)
    e1 = bool(torch.isfinite(tot)) and "awing" in parts
    old_ok = activation_of({"model": {}}) == "sigmoid"          # 老 checkpoint 无 meta
    new_ok = activation_of({"meta": {"final_activation": "relu_norm"}}) == "relu_norm"
    e_ok = e1 and old_ok and new_ok
    ok_all &= e_ok
    print(f"  softseg 前向损失 {float(tot):.6f}   损失项 {list(parts)}")
    print(f"  无 meta 的老 checkpoint 判定为 sigmoid: {old_ok}")
    print(f"  有 meta 的新 checkpoint 判定正确: {new_ok}")
    print(f"  判定: {'通过' if e_ok else '!! 异常 !!'}")

    print("\n" + "=" * W)
    print(f"总判定: {'全部通过，可以训练 SoftSeg' if ok_all else '!! 有项未通过，不要训练，把结果贴给我 !!'}")
    print("=" * W)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
