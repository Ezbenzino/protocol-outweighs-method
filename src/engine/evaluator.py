"""Evaluation: Dice/IoU/HD95/ASSD and size-stratified per-nodule Dice."""
import numpy as np
import torch

from src.metrics import metrics
from src.models.activation import to_prob


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5, activation="sigmoid"):
    model.eval()
    all_dice, all_iou = [], []   # vs union mask (all raters; strictest target)
    maj_dice = []                # vs majority vote (>= 2/4 raters; literature-comparable)
    hd95s, assds = [], []
    strat = {"micro": [], "small": [], "medium": [], "large": []}

    for sample in loader:
        image = sample["image"].to(device)
        region, _ = model(image)
        prob = to_prob(region, activation)

        for b in range(image.size(0)):
            p = (prob[b, 0] > threshold).cpu().numpy()
            gt = sample["union_mask"][b, 0].cpu().numpy().astype(bool)
            all_dice.append(metrics.dice_score(p, gt))
            all_iou.append(metrics.iou_score(p, gt))

            maj = sample["consensus"][b, 0].cpu().numpy() >= 0.5
            maj_dice.append(metrics.dice_score(p, maj))

            hd, hd95, assd = metrics.surface_distances(p, gt)
            if not np.isnan(hd):
                hd95s.append(hd95)
                assds.append(assd)

            # true per-instance dice vs majority-vote instances (nodules are now
            # defined as >=2-rater components, so GT instances come from majority)
            inst = sample["maj_instance_ids"][b].cpu().numpy()
            dia = float(sample["diameter_mm"][b].item())
            for iid in np.unique(inst):
                if iid == 0:
                    continue
                strat[metrics.size_bucket(dia)].append(
                    metrics.instance_dice(p, inst == iid))

    res = {
        "dice": float(np.mean(all_dice)) if all_dice else 0.0,
        "dice_majority": float(np.mean(maj_dice)) if maj_dice else 0.0,
        "iou": float(np.mean(all_iou)) if all_iou else 0.0,
        "hd95": float(np.mean(hd95s)) if hd95s else float("nan"),
        "assd": float(np.mean(assds)) if assds else float("nan"),
    }
    for k, v in strat.items():
        res[f"dice_{k}"] = float(np.mean(v)) if v else float("nan")

    model.train()
    return res
