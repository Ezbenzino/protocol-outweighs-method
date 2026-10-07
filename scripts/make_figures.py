# [2026-10-03] 已被 scripts/make_figures_v22.py 取代：论文 v22 的 9 张插图统一由该脚本生成；本脚本仅作历史保留，其输出不再用于论文。
"""Generate paper-quality figures for the JBHI manuscript.

1) Segmentation overlay grid: 3 size bins x 3 representative cases, each showing
   the CT patch with GT (majority-vote) contour in red and prediction contour in
   green, annotated with the per-case Dice.
2) Training curves: train loss + val Dice (majority) vs epoch, parsed from logs.
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.dataset import LungNoduleDataset
from src.engine.evaluator import evaluate  # noqa: F401  (kept for import safety)
from src.metrics import metrics
from src.models.segmenter import Segmenter
from src.utils.config import load_config

BINS = [("Micro (3-5 mm)", 0, 5), ("Small (5-10 mm)", 5, 10), ("Medium (10-30 mm)", 10, 30)]


def collect_cases(ds, model, device, per_bin=3):
    model.eval()
    cases = {name: [] for name, _, _ in BINS}
    with torch.no_grad():
        for i in range(len(ds)):
            s = ds[i]
            dia = float(s["diameter_mm"])
            name = next((n for n, lo, hi in BINS if lo <= dia < hi), None)
            if name is None:
                continue
            img = s["image"].unsqueeze(0).to(device)
            region, _ = model(img)
            pred = (torch.sigmoid(region)[0, 0] > 0.5).cpu().numpy()
            maj = (s["majority_mask"][0].cpu().numpy() > 0.5)
            d = metrics.dice_score(pred, maj)
            cases[name].append((s["image"][0].cpu().numpy(), maj, pred, dia, d))
    picked = {}
    for name, _, _ in BINS:
        arr = cases[name]
        if not arr:
            picked[name] = []
            continue
        arr_sorted = sorted(arr, key=lambda x: x[4])  # sort by dice ascending
        n = len(arr_sorted)
        idxs = sorted(set([n // 4, n // 2, 3 * n // 4]))[:per_bin]
        picked[name] = [arr_sorted[j] for j in idxs]
    return picked


def draw_overlay(ax, img, gt, pred, dia, dice):
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    if gt.sum() > 0:
        ax.contour(gt, levels=[0.5], colors="red", linewidths=1.2)
    if pred.sum() > 0:
        ax.contour(pred, levels=[0.5], colors="lime", linewidths=1.2)
    ax.set_title(f"{dia:.1f} mm | Dice={dice:.2f}", fontsize=8)
    ax.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="outputs/figures")
    ap.add_argument("--train-log", default="outputs/runs/main/logs/train.log")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    ds = LungNoduleDataset(cfg.data.npz_dir,
                           os.path.join(cfg.data.split_dir, "val.csv"), cfg, phase="val")
    model = Segmenter(backbone=cfg.model.backbone, in_channels=cfg.model.in_channels,
                      pretrained=False, num_classes=cfg.model.num_classes)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.to(device).eval()

    picked = collect_cases(ds, model, device)

    fig, axes = plt.subplots(3, 3, figsize=(7.5, 7.5))
    for r, (name, _, _) in enumerate(BINS):
        axes[r, 0].set_ylabel(name, fontsize=9)
        for c in range(3):
            ax = axes[r, c]
            if c < len(picked[name]):
                img, gt, pred, dia, d = picked[name][c]
                draw_overlay(ax, img, gt, pred, dia, d)
            else:
                ax.axis("off")
    fig.suptitle("Segmentation examples: red = GT (majority), green = prediction",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(args.out, "fig_segmentation.png"), dpi=300,
                bbox_inches="tight")
    plt.close(fig)
    print(f"saved segmentation grid -> {args.out}/fig_segmentation.png")

    # ---- training curves ----
    epochs, train_loss, val_dice, val_maj = [], [], [], []
    if os.path.exists(args.train_log):
        for line in open(args.train_log, encoding="utf-8"):
            m = re.search(r"Epoch (\d+)/\d+ \| train_loss=([\d.]+)", line)
            if m:
                epochs.append(int(m.group(1)))
                train_loss.append(float(m.group(2)))
            m = re.search(r"val \| dice=[\d.]+, dice_majority=([\d.]+)", line)
            if m and len(val_maj) < len(epochs):
                val_maj.append(float(m.group(1)))
        fig, ax1 = plt.subplots(figsize=(6, 3.6))
        ax1.plot(epochs, train_loss, "b-", label="train loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Train loss", color="b")
        ax2 = ax1.twinx()
        if val_maj:
            ax2.plot(epochs[:len(val_maj)], val_maj, "r--", label="val Dice (majority)")
            ax2.set_ylim(0.5, 1.0)
        ax2.set_ylabel("Dice", color="r")
        ax1.legend(loc="upper left", fontsize=8)
        ax2.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "fig_curves.png"), dpi=300,
                    bbox_inches="tight")
        plt.close(fig)
        print(f"saved training curves -> {args.out}/fig_curves.png")


if __name__ == "__main__":
    main()
