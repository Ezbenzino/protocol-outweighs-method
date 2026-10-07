"""Encoder backbones exposing multi-scale skip features.

Supported: resnet18 / resnet34 / convnext_tiny / pvt_v2_b1 / pvt_v2_b2 (via timm).
Each encoder exposes:
  .skip_channels : list of channel counts (shallow -> deep)
  .final_scale   : scale factor needed to upsample the shallowest feature to full res
  forward(x) -> [f1, f2, ...] (shallow -> deep)

Deepest feature is at 1/16 (never 1/32) to preserve small-nodule geometry.
"""
import torch.nn as nn
import torchvision


class ResNetEncoder(nn.Module):
    def __init__(self, name, in_channels=3, pretrained=True):
        super().__init__()
        weights_enum = {
            "resnet18": torchvision.models.ResNet18_Weights.IMAGENET1K_V1,
            "resnet34": torchvision.models.ResNet34_Weights.IMAGENET1K_V1,
        }[name]
        weights = weights_enum if pretrained else None
        resnet = getattr(torchvision.models, name)(weights=weights)

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        if in_channels != 3:
            self.stem[0] = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3  # deepest kept: 1/16 (layer4 -> 1/32 is dropped)

        self.skip_channels = [64, 64, 128, 256]
        self.final_scale = 2

    def forward(self, x):
        f1 = self.stem(x)                    # 64 @ 1/2
        f2 = self.layer1(self.maxpool(f1))   # 64 @ 1/4
        f3 = self.layer2(f2)                 # 128 @ 1/8
        f4 = self.layer3(f3)                 # 256 @ 1/16
        return [f1, f2, f3, f4]


class ConvNeXtEncoder(nn.Module):
    def __init__(self, name, in_channels=3, pretrained=True):
        super().__init__()
        weights_enum = {
            "convnext_tiny": torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
        }[name]
        weights = weights_enum if pretrained else None
        cn = getattr(torchvision.models, name)(weights=weights)

        # Keep all feature blocks as a flat ModuleList. ConvNeXt's stem is stride 4
        # (no 1/2 feature), so we detect stages by spatial downsampling instead of
        # relying on fragile `cn.features[i]` indices (which vary across versions).
        self.stages = nn.ModuleList(cn.features)
        if in_channels != 3:
            # stem is ConvNormAct (nn.Sequential) with the conv at index 0
            self.stages[0][0] = nn.Conv2d(in_channels, 96, kernel_size=4, stride=4)

        self.skip_channels = [96, 192, 384]   # 1/4, 1/8, 1/16
        self.final_scale = 4

    def forward(self, x):
        feats = []
        prev = x.shape[-1]
        for layer in self.stages:
            x = layer(x)
            if x.shape[-1] < prev:   # spatial downsampling => a new skip level
                feats.append(x)
                prev = x.shape[-1]
        return feats[:3]  # keep [1/4, 1/8, 1/16], drop 1/32


class TimmEncoder(nn.Module):
    """timm 骨干的适配层，用来把 Transformer 编码器接进同一套解码器。

    【为什么需要它】
    论文主张"协议效应 > 架构效应"，而架构效应的极差此前只来自三个同族 CNN
    U-Net（PlainUNet / ResNet-18 / ResNet-34）。审稿人可以一句话反驳：被比较的
    架构本身差异就小，极差小是必然的。论点强度取决于架构轴的【张成】，
    所以必须接进一个真正不同族的编码器。

    【为什么是 PVT v2 而不是 Swin】
    实测（scripts/probe_timm.py）：Swin-T 与 MaxViT 都把输入分辨率写死在 224/256，
    128x128 输入直接断言失败；PVT v2 用重叠卷积式 patch embedding + 空间缩减注意力，
    分辨率无关，128x128 原生可跑。PVT v2 也正是 SegFormer 的编码器家族。

    【与其它编码器保持一致的地方】
    只取 1/4, 1/8, 1/16 三级，丢掉 1/32——与 ConvNeXt 编码器相同，也与 ResNet
    编码器丢掉 layer4 的理由相同：保留小结节的几何。解码器因此完全不用改。
    """

    def __init__(self, name, in_channels=3, pretrained=True):
        super().__init__()
        import timm
        self.body = timm.create_model(
            name, features_only=True, pretrained=pretrained,
            in_chans=in_channels, out_indices=(0, 1, 2))   # 1/4, 1/8, 1/16
        self.skip_channels = list(self.body.feature_info.channels())
        self.final_scale = int(self.body.feature_info.reduction()[0])

    def forward(self, x):
        return list(self.body(x))


TIMM_BACKBONES = ("pvt_v2_b0", "pvt_v2_b1", "pvt_v2_b2", "pvt_v2_b3")


def build_encoder(backbone, in_channels=3, pretrained=True):
    if backbone in ("resnet18", "resnet34"):
        return ResNetEncoder(backbone, in_channels, pretrained)
    if backbone == "convnext_tiny":
        return ConvNeXtEncoder(backbone, in_channels, pretrained)
    if backbone in TIMM_BACKBONES or backbone.startswith("timm:"):
        return TimmEncoder(backbone.replace("timm:", ""), in_channels, pretrained)
    raise ValueError(f"Unsupported backbone: {backbone}")


def freeze_stages(encoder, stages):
    """Freeze named submodules of the encoder (e.g. ['stem', 'layer1'])."""
    for name in stages:
        if hasattr(encoder, name):
            mod = getattr(encoder, name)
            for p in mod.parameters():
                p.requires_grad = False
