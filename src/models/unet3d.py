"""Lightweight 3D U-Net for pulmonary nodule segmentation.

Designed for 8GB VRAM GPUs: base channels 32, 4 resolution levels.
Input: (B, C, D, H, W) with D=16, H=W=128.
Output: (B, 1, D, H, W) region logits.

Only Dice + BCE losses are used with this model (3D versions of
instance/boundary losses are not implemented to keep memory low).
"""
import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    """Double 3x3x3 conv + BN + ReLU."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet3D(nn.Module):
    """3D U-Net. Returns (region_logits, None) to match trainer interface."""

    def __init__(self, in_channels=1, num_classes=1, base_channels=32):
        super().__init__()
        c = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]

        # Encoder
        self.enc1 = ConvBlock3D(in_channels, c[0])
        self.enc2 = ConvBlock3D(c[0], c[1])
        self.enc3 = ConvBlock3D(c[1], c[2])
        self.enc4 = ConvBlock3D(c[2], c[3])
        self.pool = nn.MaxPool3d(2)

        # Decoder
        self.up3 = nn.ConvTranspose3d(c[3], c[2], 2, stride=2)
        self.dec3 = ConvBlock3D(c[3], c[2])
        self.up2 = nn.ConvTranspose3d(c[2], c[1], 2, stride=2)
        self.dec2 = ConvBlock3D(c[2], c[1])
        self.up1 = nn.ConvTranspose3d(c[1], c[0], 2, stride=2)
        self.dec1 = ConvBlock3D(c[1], c[0])

        self.out_conv = nn.Conv3d(c[0], num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        region = self.out_conv(d1)
        return region, None  # (region_logits, boundary_logits=None)
