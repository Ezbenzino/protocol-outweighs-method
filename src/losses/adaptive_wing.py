# -*- coding: utf-8 -*-
"""Adaptive Wing 损失（Wang et al., ICCV 2019），SoftSeg 采用的逐像素回归损失。

【为什么不是 Dice】
最初我以为 SoftSeg = 软 GT + 软 Dice，这是错的。核对原文后确认：
SoftSeg（Gros et al., Medical Image Analysis 2021）的三个组成部分是
  1) 不二值化的软 GT
  2) 归一化 ReLU 作为最终活化（见 src/models/activation.py）
  3) **Adaptive Wing 回归损失**——Dice 只出现在它的消融里
所以照"软 Dice"实现就不是 SoftSeg，比较也就不成立。

【这个损失在做什么】
它是为热图回归设计的：对【接近 0 的背景像素】用近似 L1 的惩罚，避免海量背景
主导梯度；对【取值中间的前景/边界像素】用对数型惩罚，使小误差也有较大梯度。
指数 alpha - y 让惩罚形状随目标值 y 自适应——这正是"Adaptive"的含义。

    |y - p| < theta :  omega * ln(1 + |(y-p)/eps| ** (alpha - y))
    否则            :  A * |y - p| - C
其中 A、C 由连续性与一阶可导条件解出，见下方代码。

默认超参取原文：alpha=2.1, omega=14, epsilon=1, theta=0.5。
"""
import torch


def adaptive_wing_loss(pred, target, alpha=2.1, omega=14.0, epsilon=1.0, theta=0.5):
    """pred 与 target 都是 [0,1] 的概率图，形状相同。

    注意 pred 是【已活化】的概率，不是 logits——这与本项目其它损失不同，
    因为 SoftSeg 的活化方式本身就是它的组成部分之一。
    """
    d = (target - pred).abs()
    ay = alpha - target                                   # 逐像素的自适应指数
    te = theta / epsilon
    # A 与 C 由分段处连续且一阶可导解出
    A = omega * (1.0 / (1.0 + te ** ay)) * ay * (te ** (ay - 1.0)) / epsilon
    C = theta * A - omega * torch.log1p(te ** ay)
    small = omega * torch.log1p((d / epsilon) ** ay)
    large = A * d - C
    return torch.where(d < theta, small, large).mean()
