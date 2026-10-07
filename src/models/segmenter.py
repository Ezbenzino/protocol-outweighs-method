"""Dual-branch segmentation model: region logits + boundary logits."""
import torch.nn as nn

from src.models.decoder import Decoder
from src.models.encoder import build_encoder, freeze_stages


class Segmenter(nn.Module):
    def __init__(self, backbone="resnet34", in_channels=3, pretrained=True,
                 num_classes=1, freeze=None):
        super().__init__()
        self.encoder = build_encoder(backbone, in_channels, pretrained)
        self.decoder = Decoder(
            self.encoder.skip_channels, out_channels=64, final_scale=self.encoder.final_scale)
        self.region_head = nn.Conv2d(64, num_classes, 1)
        self.boundary_head = nn.Conv2d(64, num_classes, 1)
        if freeze:
            freeze_stages(self.encoder, freeze)

    def forward(self, x):
        feats = self.encoder(x)
        d0 = self.decoder(feats)
        region = self.region_head(d0)
        boundary = self.boundary_head(d0)
        return region, boundary
