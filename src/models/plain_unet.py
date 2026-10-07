"""Standard 2D U-Net without pretrained encoder.

A plain U-Net (Ronneberger et al., 2015) with no ImageNet pretraining,
used as a baseline to isolate the effect of (a) pretrained encoders and
(b) the consensus-probability supervision target.

Architecture: 4-level encoder-decoder with skip connections,
base channels = 64, double-conv blocks, nearest-neighbor upsampling.
Single output head (region logits only; no boundary branch).
"""
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


class PlainUNet(nn.Module):
    """Standard U-Net. Returns (region_logits, None) to match Segmenter interface."""

    def __init__(self, in_channels=3, num_classes=1, base_channels=64):
        super().__init__()
        c = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]

        # Encoder
        self.enc1 = DoubleConv(in_channels, c[0])
        self.enc2 = DoubleConv(c[0], c[1])
        self.enc3 = DoubleConv(c[1], c[2])
        self.enc4 = DoubleConv(c[2], c[3])
        self.pool = nn.MaxPool2d(2)

        # Decoder
        self.up3 = nn.ConvTranspose2d(c[3], c[2], 2, stride=2)
        self.dec3 = DoubleConv(c[3], c[2])
        self.up2 = nn.ConvTranspose2d(c[2], c[1], 2, stride=2)
        self.dec2 = DoubleConv(c[2], c[1])
        self.up1 = nn.ConvTranspose2d(c[1], c[0], 2, stride=2)
        self.dec1 = DoubleConv(c[1], c[0])

        self.out_conv = nn.Conv2d(c[0], num_classes, 1)

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
