"""U-Net style decoder with bilinear upsampling and skip concatenation."""
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class Decoder(nn.Module):
    """Generic decoder over a list of skip features (shallow -> deep)."""

    def __init__(self, skip_channels, out_channels=64, final_scale=2):
        super().__init__()
        self.blocks = nn.ModuleList()
        in_ch = skip_channels[-1]
        for skip_ch in reversed(skip_channels[:-1]):
            self.blocks.append(DecoderBlock(in_ch, skip_ch, skip_ch))
            in_ch = skip_ch
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=final_scale, mode="bilinear", align_corners=False),
            DoubleConv(in_ch, out_channels),
        )

    def forward(self, feats):
        x = feats[-1]
        for i, block in enumerate(self.blocks):
            x = block(x, feats[-2 - i])
        return self.final(x)
