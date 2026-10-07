"""SITL: Scale-Invariant Target Loss.

Computes a soft Dice for every connected-component instance independently, then
averages with equal weight per instance, so each nodule contributes equally
regardless of its area (removes small-target supervision dilution).
"""
import torch


def sitl_loss(region_prob, instance_ids, smooth=1.0, min_instance_size=1):
    """region_prob: (B, 1, H, W) probs; instance_ids: (B, H, W) long, 0 = background."""
    B = region_prob.size(0)
    losses = []
    for b in range(B):
        pred = region_prob[b, 0].reshape(-1)
        ids = instance_ids[b].reshape(-1).long()
        if ids.numel() == 0:
            losses.append(torch.zeros((), device=pred.device))
            continue
        max_id = int(ids.max().item())
        if max_id <= 0:
            losses.append(torch.zeros((), device=pred.device))
            continue

        inst_sum = torch.zeros(max_id + 1, device=pred.device).scatter_add(0, ids, pred)
        inst_cnt = torch.zeros(max_id + 1, device=pred.device).scatter_add(
            0, ids, torch.ones_like(pred))
        idx = torch.arange(max_id + 1, device=pred.device)
        valid = (idx > 0) & (inst_cnt >= min_instance_size)
        if valid.sum() == 0:
            losses.append(torch.zeros((), device=pred.device))
            continue

        p = inst_sum[valid]
        c = inst_cnt[valid]
        dice = (2.0 * p + smooth) / (c + p + smooth)
        losses.append((1.0 - dice).mean())
    return torch.stack(losses).mean()
