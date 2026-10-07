"""Sweep binarization thresholds on a split using a trained checkpoint.

Zero retraining cost: picks the threshold that maximizes the target metric
on val, to be reported alongside the default 0.5 on test.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from src.data.dataset import LungNoduleDataset
from src.engine.evaluator import evaluate
from src.models.segmenter import Segmenter
from src.utils.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--thresholds", default="0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70")
    ap.add_argument("--metric", default="dice", help="metric to maximize")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = LungNoduleDataset(cfg.data.npz_dir,
                           os.path.join(cfg.data.split_dir, f"{args.split}.csv"),
                           cfg, phase="val")
    loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=False,
                        num_workers=cfg.data.num_workers)

    model = Segmenter(backbone=cfg.model.backbone, in_channels=cfg.model.in_channels,
                      pretrained=False, num_classes=cfg.model.num_classes)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.to(device)

    rows = []
    for t in [float(x) for x in args.thresholds.split(",")]:
        res = evaluate(model, loader, device, t)
        rows.append((t, res))
        print(f"thr={t:.2f} " + " ".join(f"{k}={v:.4f}" for k, v in res.items()
                                         if isinstance(v, float)), flush=True)

    best_t, best_res = max(rows, key=lambda r: r[1].get(args.metric, 0.0))
    print(f"\nbest threshold = {best_t:.2f} ({args.metric}={best_res[args.metric]:.4f})")

    out = {
        "split": args.split,
        "metric": args.metric,
        "best_threshold": best_t,
        "best": best_res,
        "all": {f"{t:.2f}": r for t, r in rows},
    }
    out_path = os.path.join(os.path.dirname(args.ckpt), f"threshold_sweep_{args.split}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
