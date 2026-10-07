"""Loss sanity tests (shape / value range / perfect prediction)."""
import torch

from src.losses.brbc import brbc_loss
from src.losses.csl import csl_loss
from src.losses.dice import soft_dice_loss
from src.losses.lovasz import lovasz_hinge
from src.losses.sitl import sitl_loss


def test_soft_dice_perfect():
    pred = torch.ones(1, 1, 8, 8)
    tgt = torch.ones(1, 1, 8, 8)
    assert soft_dice_loss(pred, tgt).item() < 1e-3


def test_sitl_range():
    pred = torch.rand(1, 1, 16, 16)
    ids = torch.zeros(16, 16, dtype=torch.long)
    ids[2:6, 2:6] = 1
    ids[10:14, 10:14] = 2
    v = sitl_loss(pred, ids.unsqueeze(0))
    assert 0.0 <= v.item() <= 1.0


def test_csl_scalar():
    logits = torch.randn(2, 1, 16, 16)
    c = torch.rand(2, 1, 16, 16)
    d = torch.rand(2, 1, 16, 16)
    assert csl_loss(logits, c, d).dim() == 0


def test_brbc_scalar():
    r = torch.randn(2, 1, 32, 32)
    b = torch.randn(2, 1, 32, 32)
    assert brbc_loss(r, b).dim() == 0


def test_lovasz_perfect():
    logits = torch.ones(2, 1, 8, 8) * 10
    tgt = torch.ones(2, 1, 8, 8)
    assert lovasz_hinge(logits, tgt).item() < 1e-2
