"""Lovász-Hinge loss (binary), small-object friendly.

Reference: Berman, Triki, Blaschko, "The Lovász-Softmax loss" (CVPR 2018).
"""
import torch
import torch.nn.functional as F


def lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge_flat(logits, labels):
    if labels.numel() == 0:
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted).detach()
    return torch.dot(F.relu(errors_sorted), grad)


def lovasz_hinge(logits, labels, per_image=True):
    """logits: (B, 1, H, W) raw logits; labels: (B, 1, H, W) binary."""
    if per_image:
        return torch.mean(torch.stack([
            lovasz_hinge_flat(logits[i].reshape(-1), labels[i].reshape(-1))
            for i in range(logits.size(0))
        ]))
    return lovasz_hinge_flat(logits.reshape(-1), labels.reshape(-1))
