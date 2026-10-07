"""Combined loss with configurable weights and ablation switches.

【本次重写要解决的问题】
旧版有一个会污染核心对照实验的 bug：CSL 无条件读 sample["consensus"]，
即使 loss.target 设成 union 也一样。结果是所有"并集"实验臂都偷偷吃到了
共识概率监督，不是纯并集模型，核心对比的对照组不干净。

新版把"训练靶"抽象成一张规格表 TARGET_SPECS，每个损失项从表里取自己的靶，
不再有任何硬编码。这样加新的训练靶只需要在表里加一行。

【四个训练靶】
  union      : Dice/BCE 都用并集掩膜 V>=1              —— 并集基线
  majority   : Dice/BCE 都用多数投票掩膜 V>=2          —— 拆解混淆变量的关键臂
  consensus  : Dice 用 V>=2 硬掩膜，BCE 用软概率 p=V/4 —— 原 main 的干净版
  soft       : Dice 和 BCE 都用软概率 p=V/4           —— 纯概率监督
  svls       : Dice 用 V>=2 硬掩膜，BCE 用【高斯平滑出来的】软靶
               —— 文献对照臂（Islam & Glocker, IPMI 2021 的再实现）。
                  与 consensus 臂形式完全相同，唯一差别是软监督来自几何平滑
                  还是来自医生间的真实分歧。见 src/losses/svls.py 的说明。

majority 这一臂是必须的：union 和 consensus 之间同时差了"掩膜大小定义"
和"软/硬监督"两件事，没有 majority 就无法归因。
"""
import torch
import torch.nn as nn

from src.losses.brbc import brbc_loss
from src.losses.csl import csl_loss
from src.losses.dice import bce_with_logits, soft_dice_loss
from src.losses.lovasz import lovasz_hinge
from src.losses.sitl import sitl_loss
from src.losses.svls import gaussian_kernel_2d, svls_target
from src.losses.adaptive_wing import adaptive_wing_loss
from src.models.activation import to_prob

# 每个训练靶下，各损失项分别用哪个监督信号。
#   dice / bce : sample 字典里的键
#   binary     : Lovász 这类必须要硬掩膜的损失用哪个（软靶模式下退回多数投票）
#   inst / bnd : SITL 和 BRBC 的实例图与边界图，必须跟训练靶一致
TARGET_SPECS = {
    "union": dict(dice="union_mask", bce="union_mask", binary="union_mask",
                  inst="instance_ids", bnd="boundary"),
    "majority": dict(dice="majority_mask", bce="majority_mask", binary="majority_mask",
                     inst="maj_instance_ids", bnd="maj_boundary"),
    "consensus": dict(dice="majority_mask", bce="consensus", binary="majority_mask",
                      inst="maj_instance_ids", bnd="maj_boundary"),
    "soft": dict(dice="consensus", bce="consensus", binary="majority_mask",
                 inst="maj_instance_ids", bnd="maj_boundary"),
    # "@svls" 前缀表示这个靶不在 sample 字典里，需要现场从别的靶推导出来。
    "svls": dict(dice="majority_mask", bce="@svls", binary="majority_mask",
                 inst="maj_instance_ids", bnd="maj_boundary"),
    # SoftSeg：软 GT + 归一化 ReLU 活化 + Adaptive Wing 回归损失。
    # 它同时改了靶、活化、损失族三样东西，所以【不属于】本文的受控因子表，
    # 只能单列成"与已发表方法的对照"。见 src/models/activation.py 的说明。
    "softseg": dict(dice="consensus", bce="consensus", binary="majority_mask",
                    awing="consensus", inst="maj_instance_ids", bnd="maj_boundary"),
}


class CombinedLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.lc = cfg.loss
        self.target_mode = getattr(cfg.loss, "target", "consensus")
        if self.target_mode not in TARGET_SPECS:
            raise ValueError(
                f"未知的 loss.target={self.target_mode!r}，可选：{list(TARGET_SPECS)}")
        self.spec = TARGET_SPECS[self.target_mode]
        self.activation = getattr(cfg.model, "final_activation", "sigmoid")
        # BCE / CSL / Lovasz 都吃 logits，只有 sigmoid 语义下才成立。
        # 换成归一化 ReLU 还打开它们，会静默算出无意义的损失——直接拦住。
        if self.activation != "sigmoid":
            bad = [n for n, on in (("bce", self.lc.use_bce), ("csl", self.lc.use_csl),
                                   ("lovasz", self.lc.use_lovasz)) if on]
            if bad:
                raise ValueError(
                    f"final_activation={self.activation} 时不能启用 {bad}："
                    f"它们直接作用于 logits，只在 sigmoid 语义下有定义。")
        # SVLS 的高斯核：尺寸与 sigma 必须能从配置读出并写进论文，不留隐藏常数
        sv = getattr(cfg.loss, "svls", None)
        self.svls_ksize = int(getattr(sv, "ksize", 3)) if sv is not None else 3
        self.svls_sigma = float(getattr(sv, "sigma", 1.0)) if sv is not None else 1.0
        # center_boost 是原文做法（中心权重 = 邻域权重和）。关掉它就退化成普通
        # 高斯核，平滑更狠，等于把对照方法做残——只应在消融里用。
        self.svls_center_boost = bool(getattr(sv, "center_boost", True)) if sv is not None else True
        self.register_buffer("svls_kernel",
                             gaussian_kernel_2d(self.svls_ksize, self.svls_sigma,
                                                center_boost=self.svls_center_boost),
                             persistent=False)

    def forward(self, region_logits, boundary_logits, sample):
        region_prob = to_prob(region_logits, self.activation)
        s = self.spec

        def fetch(key):
            """取靶。"@" 开头的是需要现场推导的派生靶，其余直接查 sample。"""
            if key == "@svls":
                return svls_target(sample["majority_mask"], self.svls_kernel)
            return sample[key]

        dice_target = fetch(s["dice"])       # 可能是软概率，soft_dice_loss 支持
        bce_target = fetch(s["bce"])         # 可能是软概率，BCE-with-logits 支持
        binary_target = fetch(s["binary"])   # Lovász 必须要 0/1

        losses = {}
        if self.lc.use_dice:
            losses["dice"] = self.lc.weights.dice * soft_dice_loss(region_prob, dice_target)
        if self.lc.use_bce:
            losses["bce"] = self.lc.weights.bce * bce_with_logits(region_logits, bce_target)
        if self.lc.use_sitl:
            losses["sitl"] = self.lc.weights.sitl * sitl_loss(
                region_prob, sample[s["inst"]],
                min_instance_size=self.lc.sitl.min_instance_size)
        if self.lc.use_csl:
            # 关键修复：CSL 的靶跟随训练靶，不再无条件用 consensus。
            # 旧版在 union 模式下也传 sample["consensus"]，导致并集臂被共识监督污染。
            losses["csl"] = self.lc.weights.csl * csl_loss(
                region_logits, bce_target, sample["disagreement"],
                beta=self.lc.csl.beta)
        if self.lc.use_brbc:
            losses["brbc"] = self.lc.weights.brbc * brbc_loss(
                region_logits, boundary_logits,
                boundary_gt=sample.get(s["bnd"]),
                r2b_weight=self.lc.brbc.r2b_weight,
                b2r_weight=self.lc.brbc.b2r_weight,
                boundary_gt_weight=self.lc.brbc.boundary_gt_weight)
        if getattr(self.lc, "use_awing", False):
            # AWing 吃【已活化】的概率，不是 logits——SoftSeg 的活化本身是方法的一部分
            losses["awing"] = getattr(self.lc.weights, "awing", 1.0) * adaptive_wing_loss(
                region_prob, fetch(s.get("awing", "consensus")),
                alpha=float(getattr(getattr(self.lc, "awing", None), "alpha", 2.1)),
                omega=float(getattr(getattr(self.lc, "awing", None), "omega", 14.0)),
                epsilon=float(getattr(getattr(self.lc, "awing", None), "epsilon", 1.0)),
                theta=float(getattr(getattr(self.lc, "awing", None), "theta", 0.5)))
        if self.lc.use_lovasz:
            losses["lovasz"] = self.lc.weights.lovasz * lovasz_hinge(region_logits, binary_target)

        total = sum(losses.values())
        return total, {k: v.detach() for k, v in losses.items()}
