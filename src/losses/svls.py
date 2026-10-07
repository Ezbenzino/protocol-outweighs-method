# -*- coding: utf-8 -*-
"""SVLS 式空间可变标签平滑（Islam & Glocker, IPMI 2021）的靶构造。

【为什么这个对照臂重要】
本文的 C 臂（consensus）用 Dice 打硬掩膜 V>=2、用 BCE 打软概率 p=V/4，
软监督来自【四位医生的真实投票】。SVLS 在形式上完全一样——硬掩膜打 Dice、
一张软标签打逐像素损失——但那张软标签是【从单张融合掩膜的几何形状】推出来的，
用一个固定核在边界附近造出渐变，与标注者间的实际分歧无关。

所以 SVLS 与 C 臂构成一个很干净的对照：**唯一的差别是软监督来自"真实分歧"
还是"几何平滑"**。若 C 臂相对 SVLS 没有优势，就说明多标注者投票带来的信息，
用一个不需要多标注者的固定核即可等价替代——这对整条"共识概率监督"的文献线
是一个有分量的阴性结果。审稿人会问这个问题，不如自己先答。

【核的构造：不是普通高斯核】
原文的核在普通高斯核基础上多做一步——把【中心权重替换为邻域权重之和】，
再整体缩放使中心权重为 1：

    K = 归一化高斯核（和为 1）
    s = 1 - K[中心]                    # 邻域权重之和
    K[中心] = s
    K = K / s                          # 中心变成 1，此时 sum(K) == 2

这一步的作用是抬高像素自身标签的权重，让病灶内部保持接近 1、边界少被拉扯。
**若照普通高斯核实现，平滑会比原文更狠，等于把对照方法做残了**，
比较就不公平。二分类单通道下，多类版本"逐通道卷积后跨通道归一化"等价于
除以 sum(K)，所以此处直接除以 sum(K)。

【与原文的关系（必须在论文里如实写明）】
这是**再实现**，不是原作者代码。原文面向多类分割、逐像素损失用软标签交叉熵；
本文是二分类单 logit 设定，对应位置用 BCE。原文 sigma = 1；
**原文正文未写明核尺寸**（官方实现默认 3），也未做核尺寸/sigma 的消融。
本文的核尺寸按 scripts/calibrate_svls.py 在本队列上实测的标注者分歧带宽标定，
并在配置里显式给出。不声称与原作者代码逐位一致。

【边界填充为什么用 replicate】
零填充会让 patch 四边被人为压暗，模型可能去学"靠近图像边缘标签更弱"——
那是一条只在训练时存在的伪线索。本项目已经因为"训练时才有的伪影"吃过亏
（位置先验那一节），这里不重蹈覆辙。
"""
import torch
import torch.nn.functional as F


def gaussian_kernel_2d(ksize: int, sigma: float, center_boost: bool = True,
                       device=None, dtype=torch.float32):
    """构造 SVLS 核，形状 (1, 1, k, k)。

    center_boost=True  按原文：中心权重 = 邻域权重和，再缩放使中心为 1（sum(K)=2）
    center_boost=False 普通归一化高斯核（sum(K)=1）——仅用于消融对照，不是原文做法
    """
    if ksize % 2 == 0:
        raise ValueError(f"SVLS 核尺寸必须是奇数，收到 {ksize}")
    ax = torch.arange(ksize, device=device, dtype=dtype) - (ksize - 1) / 2.0
    g = torch.exp(-(ax ** 2) / (2.0 * sigma ** 2))
    k = torch.outer(g, g)
    k = k / k.sum()                                   # 归一化高斯核
    if center_boost:
        c = ksize // 2
        s = 1.0 - k[c, c]                             # 邻域权重之和
        if float(s) <= 0:
            raise ValueError(
                f"ksize={ksize} sigma={sigma} 下中心权重已占满整个核，"
                f"邻域权重和为 {float(s):.3e}，核退化成 delta。请增大 sigma 或 ksize。")
        k = k.clone()
        k[c, c] = s
        k = k / s                                     # 中心 -> 1，此时 k.sum() == 2
    return k.view(1, 1, ksize, ksize)


def svls_target(mask: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """把二值掩膜 (B,1,H,W) 平滑成 [0,1] 软靶。靶不参与反向传播。

    除以 kernel.sum() 而不是假设核已归一化：center_boost 的核和为 2，
    普通高斯核和为 1，这样两种核都能得到正确的 [0,1] 取值。
    """
    if mask.dim() != 4 or mask.shape[1] != 1:
        raise ValueError(f"期望 (B,1,H,W) 的掩膜，收到 {tuple(mask.shape)}")
    p = kernel.shape[-1] // 2
    ker = kernel.to(mask.dtype).to(mask.device)
    x = F.pad(mask.detach(), (p, p, p, p), mode="replicate")
    return (F.conv2d(x, ker) / ker.sum()).clamp_(0.0, 1.0)
