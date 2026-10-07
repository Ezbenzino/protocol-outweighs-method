"""3D U-Net training entry point.

Independent from scripts/train.py because:
- Uses LungNoduleDataset3D (volumetric patches)
- Uses UNet3D model
- Uses only Dice + BCE losses (3D-safe; auxiliary losses not ported)
- Evaluation computes Dice per-sample (HD95/ASSD computed per-slice for compatibility)

Usage:
    python scripts/train_3d.py --config configs/unet3d.yaml --name unet3d
"""
import argparse
import gc
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset3d import LungNoduleDataset3D
from src.models.unet3d import UNet3D
from src.losses.dice import soft_dice_loss, bce_with_logits
from src.utils.config import load_config
from src.utils.seed import set_seed


def dice_score_np(pred, gt):
    """Binary Dice for numpy arrays."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = (pred & gt).sum()
    union = pred.sum() + gt.sum()
    if union == 0:
        return 1.0
    return 2.0 * intersection / union


@torch.no_grad()
def evaluate_3d(model, loader, device, threshold=0.5):
    """Evaluate 3D U-Net. Computes Dice vs majority and union masks."""
    model.eval()
    maj_dices, union_dices = [], []
    micro_dices, small_dices, medium_dices = [], [], []

    for sample in loader:
        image = sample["image"].to(device)
        region, _ = model(image)
        prob = torch.sigmoid(region)

        for b in range(image.size(0)):
            p = (prob[b, 0] > threshold).cpu().numpy()
            maj = sample["majority_mask"][b, 0].cpu().numpy().astype(bool)
            uni = sample["union_mask"][b, 0].cpu().numpy().astype(bool)
            dia = float(sample["diameter_mm"][b].item())

            maj_dices.append(dice_score_np(p, maj))
            union_dices.append(dice_score_np(p, uni))

            if dia < 5:
                micro_dices.append(dice_score_np(p, maj))
            elif dia < 10:
                small_dices.append(dice_score_np(p, maj))
            elif dia < 30:
                medium_dices.append(dice_score_np(p, maj))

    model.train()
    return {
        "dice_majority": float(np.mean(maj_dices)) if maj_dices else 0.0,
        "dice": float(np.mean(union_dices)) if union_dices else 0.0,
        "dice_micro": float(np.mean(micro_dices)) if micro_dices else float("nan"),
        "dice_small": float(np.mean(small_dices)) if small_dices else float("nan"),
        "dice_medium": float(np.mean(medium_dices)) if medium_dices else float("nan"),
    }


class EarlyStopping:
    def __init__(self, patience=12, mode="max"):
        self.patience = patience
        self.mode = mode
        self.best = -float("inf") if mode == "max" else float("inf")
        self.counter = 0
        self.best_epoch = 0

    def step(self, value, epoch):
        improved = (value > self.best) if self.mode == "max" else (value < self.best)
        if improved:
            self.best = value
            self.counter = 0
            self.best_epoch = epoch
            return False, True
        else:
            self.counter += 1
            return self.counter >= self.patience, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", default="unet3d")
    args = ap.parse_args()

    cfg = load_config(args.config)
    run_dir = os.path.join(cfg.outputs.root, "runs", args.name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    log_dir = os.path.join(run_dir, "logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    set_seed(cfg.project.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_ds = LungNoduleDataset3D(cfg.data.npz_dir, os.path.join(cfg.data.split_dir, "train.csv"),
                                    cfg, phase="train")
    val_ds = LungNoduleDataset3D(cfg.data.npz_dir, os.path.join(cfg.data.split_dir, "val.csv"),
                                  cfg, phase="val")
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True,
                              num_workers=cfg.data.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False,
                            num_workers=cfg.data.num_workers)

    # Model
    model = UNet3D(in_channels=cfg.model.in_channels, num_classes=cfg.model.num_classes,
                   base_channels=getattr(cfg.model, "base_channels", 32))
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                                   weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.train.epochs * len(train_loader), eta_min=1e-6)
    es = EarlyStopping(patience=cfg.train.early_stop_patience, mode="max")

    use_amp = getattr(cfg.train, "use_amp", True)
    amp_dtype = torch.bfloat16 if getattr(cfg.train, "amp_dtype", "bf16") == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    log_path = os.path.join(log_dir, "train.log")
    with open(log_path, "w") as f:
        f.write(f"train={len(train_ds)} val={len(val_ds)} device={device}\n")

    print(f"3D U-Net | train={len(train_ds)} val={len(val_ds)} device={device}")
    print(f"  batch_size={cfg.train.batch_size} depth={getattr(cfg.data, 'depth', 16)} "
          f"base_channels={getattr(cfg.model, 'base_channels', 32)}")

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0
        t0 = time.time()

        for step, sample in enumerate(train_loader):
            image = sample["image"].to(device)
            if cfg.loss.target == "consensus":
                soft_target = sample["consensus"].to(device)
                bin_target = sample["majority_mask"].to(device)
            else:
                soft_target = sample["union_mask"].to(device)
                bin_target = sample["union_mask"].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                region, _ = model(image)
                loss_dice = soft_dice_loss(torch.sigmoid(region), bin_target)
                loss_bce = bce_with_logits(region, soft_target)
                loss = loss_dice + loss_bce

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            n += 1

        train_loss = total_loss / max(1, n)
        elapsed = time.time() - t0

        # Validation
        metrics = evaluate_3d(model, val_loader, device)
        monitor = metrics["dice_majority"]
        stop, is_best = es.step(monitor, epoch)

        log_line = (f"Epoch {epoch}/{cfg.train.epochs} | loss={train_loss:.4f} | "
                    f"val_dice_maj={metrics['dice_majority']:.4f} | "
                    f"val_dice_union={metrics['dice']:.4f} | {elapsed:.1f}s")
        print(log_line)
        with open(log_path, "a") as f:
            f.write(log_line + "\n")

        if is_best:
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics},
                       os.path.join(ckpt_dir, "best.pth"))
            best_line = f"  saved best (epoch {epoch}, dice_majority={monitor:.4f})"
            print(best_line)
            with open(log_path, "a") as f:
                f.write(best_line + "\n")

        if stop:
            print(f"Early stopping at epoch {epoch}")
            break

        # Memory cleanup to prevent leaks across epochs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    torch.save({"model": model.state_dict(), "epoch": epoch},
               os.path.join(ckpt_dir, "last.pth"))
    print(f"Training complete. Best dice_majority={es.best:.4f} at epoch {es.best_epoch}")


if __name__ == "__main__":
    main()
