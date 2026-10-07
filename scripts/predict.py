"""Inference entry point: predict and visualize overlaid masks."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.data.dataset import LungNoduleDataset
from src.models.segmenter import Segmenter
from src.utils.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--num", type=int, default=4)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = LungNoduleDataset(cfg.data.npz_dir, os.path.join(cfg.data.split_dir, "val.csv"),
                           cfg, phase="val")
    model = Segmenter(backbone=cfg.model.backbone, in_channels=cfg.model.in_channels,
                      pretrained=False, num_classes=cfg.model.num_classes)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.to(device).eval()

    out_dir = cfg.outputs.prediction_dir
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        for i in range(min(args.num, len(ds))):
            sample = ds[i]
            image = sample["image"].unsqueeze(0).to(device)
            region, _ = model(image)
            pred = (torch.sigmoid(region)[0, 0] > 0.5).cpu().numpy()

            img = sample["image"][0].cpu().numpy()
            gt = sample["union_mask"][0].cpu().numpy()

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(img, cmap="gray")
            axes[0].set_title("CT")
            axes[1].imshow(gt, cmap="gray")
            axes[1].set_title("GT")
            axes[2].imshow(pred, cmap="gray")
            axes[2].set_title("Pred")
            for ax in axes:
                ax.axis("off")
            fig.savefig(os.path.join(out_dir, f"pred_{i}.png"), bbox_inches="tight")
            plt.close(fig)
    print(f"saved {min(args.num, len(ds))} overlays -> {out_dir}")


if __name__ == "__main__":
    main()
