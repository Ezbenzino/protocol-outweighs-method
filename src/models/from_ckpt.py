# -*- coding: utf-8 -*-
"""从 checkpoint 自身推断架构并建模型——【全项目唯一的一份实现】。

【为什么不看 config】
一条评测命令要同时跑多个架构臂（tgtB 是 ResNet-34、archR18 是 ResNet-18、
archPU 是 PlainUNet、archCNX 是 ConvNeXt、archPVT 是 PVT v2），它们各有自己的
config。若统一用 default.yaml 建模型，archPU 会被建成 ResNet-34 然后加载失败。
做法是逐个试候选架构，取第一个能 strict 加载成功的——不依赖任何命名约定。

【为什么必须只有一份】
这段逻辑原本在 eval_symmetric.py 和 eval_scope.py 里各抄了一份。加 PVT v2 时
只改了训练侧的 encoder.py，两个评测脚本的候选表都没同步，结果是模型训完了
才在评测时报"没有候选架构能加载该 checkpoint"。现在候选表【直接从
src/models/encoder.py 的 TIMM_BACKBONES 生成】，加新骨干时不需要改这里，
也就不会再出现改一处漏一处。
"""
from src.models.activation import activation_of, to_prob
from src.models.encoder import TIMM_BACKBONES
from src.models.plain_unet import PlainUNet
from src.models.segmenter import Segmenter

# Segmenter 支持的编码器：内建的 + encoder.py 声明的 timm 骨干
SEGMENTER_BACKBONES = ("resnet34", "resnet18", "convnext_tiny") + TIMM_BACKBONES


def build_model_for_ckpt(state_dict, cfg, activation="sigmoid"):
    """返回 (model, arch_name)。arch_name 是真正加载成功的那个架构。

    activation 会被钉在 model.final_activation 上，之后一律用 region_prob(model, x)
    取概率，调用方不需要再关心这个模型是 sigmoid 还是归一化 ReLU。
    """
    ic, nc = cfg.model.in_channels, cfg.model.num_classes
    candidates = [("plain_unet",
                   lambda: PlainUNet(in_channels=ic, num_classes=nc,
                                     base_channels=getattr(cfg.model, "base_channels", 64)))]
    for b in SEGMENTER_BACKBONES:
        candidates.append(
            (b, (lambda bb: lambda: Segmenter(backbone=bb, in_channels=ic,
                                              pretrained=False, num_classes=nc))(b)))

    # 先试 config 指定的那个，命中率最高，也省掉无谓的建模开销
    named = getattr(cfg.model, "type", None) or getattr(cfg.model, "backbone", None)
    candidates.sort(key=lambda c: c[0] != named)

    errors = []
    for name, make in candidates:
        try:
            m = make()
            m.load_state_dict(state_dict, strict=True)
            m.final_activation = activation
            return m, name
        except Exception as e:                                # noqa: BLE001
            errors.append(f"{name}: {type(e).__name__}")
    raise RuntimeError(
        "没有候选架构能加载该 checkpoint。尝试过 -> " + "; ".join(errors)
        + "\n若这是新加的架构：确认它已登记在 src/models/encoder.py 的 "
          "TIMM_BACKBONES 或 build_encoder 里，本函数的候选表由那里自动生成。")


def forward_region(model, x):
    """统一取 region 分支：双分支模型返回 (region, boundary)，单分支直接返回。"""
    out = model(x)
    return out[0] if isinstance(out, (tuple, list)) else out


def region_prob(model, x):
    """前向 + 按该模型自己的活化方式转成概率。

    评测脚本一律用这个函数，不要再直接写 torch.sigmoid ——
    SoftSeg 用的是归一化 ReLU，硬编码 sigmoid 会静默算出错误的概率图，
    而且因为数值仍在 [0,1]，不会报错，只会得到一份看似正常的错误结果。
    """
    return to_prob(forward_region(model, x), getattr(model, "final_activation", "sigmoid"))


def load_for_eval(ckpt_path, cfg, map_location="cpu"):
    """读 checkpoint -> (model, arch, activation)。活化方式来自 checkpoint 的 meta。"""
    import torch
    state = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    act = activation_of(state)
    model, arch = build_model_for_ckpt(state["model"], cfg, activation=act)
    return model, arch, act
