"""CSL: Consensus-based Soft Labeling loss.

  consensus    : vote ratio p in [0, 1] from 4 raters (soft label)
  disagreement : 4 * p * (1 - p) in [0, 1] (uncertainty)

Two terms:
  1) consensus-weighted BCE (on logits, autocast-safe): strong supervision where
     raters agree (weight = 1 - beta * disagreement)
  2) disagreement low-confidence: push prediction toward 0.5 in disputed regions
"""
import torch
import torch.nn.functional as F


def csl_loss(region_logits, consensus, disagreement, beta=0.75, eps=1e-6):
    logits = region_logits.float()
    consensus = consensus.float()
    disagreement = disagreement.float()

    w_c = 1.0 - beta * disagreement
    bce = F.binary_cross_entropy_with_logits(logits, consensus, reduction="none")
    l_consensus = (w_c * bce).mean()

    p = torch.sigmoid(logits).clamp(eps, 1.0 - eps)
    l_low_conf = (disagreement * (p - 0.5) ** 2).mean()
    return l_consensus + l_low_conf
