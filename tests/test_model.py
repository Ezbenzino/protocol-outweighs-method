"""Model shape sanity tests."""
import torch

from src.models.segmenter import Segmenter


def test_segmenter_resnet34_shapes():
    m = Segmenter(backbone="resnet34", in_channels=3, pretrained=False)
    x = torch.randn(2, 3, 128, 128)
    region, boundary = m(x)
    assert region.shape == (2, 1, 128, 128)
    assert boundary.shape == (2, 1, 128, 128)


def test_segmenter_resnet18_shapes():
    m = Segmenter(backbone="resnet18", in_channels=3, pretrained=False)
    x = torch.randn(2, 3, 128, 128)
    region, boundary = m(x)
    assert region.shape == (2, 1, 128, 128)
    assert boundary.shape == (2, 1, 128, 128)


def test_segmenter_convnext_shapes():
    m = Segmenter(backbone="convnext_tiny", in_channels=3, pretrained=False)
    x = torch.randn(2, 3, 128, 128)
    region, boundary = m(x)
    assert region.shape == (2, 1, 128, 128)
    assert boundary.shape == (2, 1, 128, 128)
