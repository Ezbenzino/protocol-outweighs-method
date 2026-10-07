"""BRBC: Boundary-Region Bidirectional Consistency.

  region -> boundary (r2b): region prediction gradient contour should match
                            the boundary prediction.
  boundary -> region (b2r): region should be sharp at boundary pixels and
                            smooth elsewhere (boundary-aware regularization).

An optional low-weight BCE against a boundary GT (morphological gradient of the
mask) anchors the boundary branch and prevents collapse.
"""
import torch
import torch.nn.functional as F


def gradient_magnitude(x):
    """Central-difference gradient magnitude of (B, 1, H, W)."""
    dx = x[..., 1:, :] - x[..., :-1, :]
    dy = x[..., :, 1:] - x[..., :, :-1]
    dx = F.pad(dx, (0, 0, 0, 1))
    dy = F.pad(dy, (0, 1, 0, 0))
    return torch.sqrt(dx * dx + dy * dy + 1e-6)


def brbc_loss(region_logits, boundary_logits, boundary_gt=None,
              r2b_weight=0.5, b2r_weight=0.5, boundary_gt_weight=0.1):
    region_prob = torch.sigmoid(region_logits)
    boundary_prob = torch.sigmoid(boundary_logits)

    grad = gradient_magnitude(region_prob)
    grad = grad / (grad.detach().max() + 1e-6)  # normalize to [0, 1]

    l_r2b = F.mse_loss(boundary_prob, grad)
    l_b2r = (boundary_prob * (1.0 - grad) + (1.0 - boundary_prob) * grad).mean()

    loss = r2b_weight * l_r2b + b2r_weight * l_b2r
    if boundary_gt is not None and boundary_gt_weight > 0:
        loss = loss + boundary_gt_weight * F.binary_cross_entropy_with_logits(
            boundary_logits, boundary_gt)
    return loss
