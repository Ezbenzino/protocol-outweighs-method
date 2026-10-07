"""Soft Dice loss and BCE."""
import torch
import torch.nn.functional as F


def soft_dice_loss(pred, target, smooth=1.0, eps=1e-6):
    """pred / target: (B, 1, H, W) probabilities in [0, 1]."""
    pred = pred.contiguous().view(pred.size(0), -1)
    target = target.contiguous().view(target.size(0), -1)
    inter = (pred * target).sum(dim=1)
    denom = pred.sum(dim=1) + target.sum(dim=1)
    dice = (2.0 * inter + smooth) / (denom + smooth + eps)
    return (1.0 - dice).mean()


def bce_with_logits(logits, target, pos_weight=None):
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
