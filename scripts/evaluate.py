"""Evaluation entry point."""
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
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = LungNoduleDataset(cfg.data.npz_dir, os.path.join(cfg.data.split_dir, f"{args.split}.csv"),
                           cfg, phase="val")
    loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=False,
                        num_workers=cfg.data.num_workers)

    model = Segmenter(backbone=cfg.model.backbone, in_channels=cfg.model.in_channels,
                      pretrained=False, num_classes=cfg.model.num_classes)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.to(device)

    res = evaluate(model, loader, device, cfg.eval.thresholds)
    print(json.dumps(res, indent=2))

    out_path = os.path.join(cfg.outputs.log_dir, f"{args.split}_metrics.json")
    os.makedirs(cfg.outputs.log_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
