# -*- coding: utf-8 -*-
"""输出活化函数的唯一入口。

【为什么需要它】
本项目所有模型的 forward 都返回未活化的 logits，活化在【模型之外】做——
训练损失、验证指标、四个评测脚本各自调 torch.sigmoid。这在只有一种活化时没问题。
SoftSeg（Gros et al., MedIA 2021）用的是"归一化 ReLU"而非 sigmoid，
于是 sigmoid 这个隐含假设就散落在六处，改一处漏一处必然出错
（本项目刚因为架构注册表分家吃过一次亏）。

现在的约定：
  - 活化方式随 checkpoint 走，存在 state["meta"]["final_activation"]
  - 读取端一律用 prob_from_ckpt(state) 拿到活化函数，不再直接写 torch.sigmoid
  - 【向后兼容】老 checkpoint 没有 meta 字段，一律按 sigmoid 处理，
    因此所有已完成实验的数值一个比特都不会变

【归一化 ReLU 是什么】
SoftSeg 的动机是 sigmoid/softmax 会把输出推向饱和的 0/1，压掉软信息。
它改用 ReLU 再按【每个样本自身的最大值】归一化，使输出落在 [0,1] 且不饱和：
    p = relu(x) / max(relu(x))
全零输出时分母为 0，此处退化为全零图（而不是 NaN）。

【一个必须知道的后果】
归一化是逐样本的，所以同一个阈值在不同样本上对应的原始响应强度不同。
这使 SoftSeg 与本文受控设计中的其它臂【输出契约不同】，不能并列进那张
受控因子表；它属于"与已发表方法的对照"，要单独成表并写明这一点。
"""
import torch

SIGMOID = "sigmoid"
RELU_NORM = "relu_norm"
VALID = (SIGMOID, RELU_NORM)


def to_prob(logits: torch.Tensor, kind: str = SIGMOID) -> torch.Tensor:
    if kind == SIGMOID:
        return torch.sigmoid(logits)
    if kind == RELU_NORM:
        r = torch.relu(logits)
        m = r.flatten(1).max(dim=1).values.view(-1, *([1] * (r.dim() - 1)))
        return r / m.clamp_min(1e-6)          # 全零样本 -> 全零图，不产生 NaN
    raise ValueError(f"未知的活化方式 {kind!r}，可选 {VALID}")


def activation_of(state: dict) -> str:
    """从 checkpoint 读活化方式。没有 meta 的老 checkpoint 一律按 sigmoid。"""
    kind = (state or {}).get("meta", {}).get("final_activation", SIGMOID)
    if kind not in VALID:
        raise ValueError(f"checkpoint 里的 final_activation={kind!r} 不认识")
    return kind
