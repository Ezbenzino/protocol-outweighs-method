"""Training entry point."""
import argparse
import os
import sys

# allow running as `python scripts/train.py` from anywhere
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from src.data.dataset import LungNoduleDataset
from src.engine.trainer import Trainer
from src.losses.combo import CombinedLoss
from src.models.segmenter import Segmenter
from src.models.plain_unet import PlainUNet
from src.utils.callbacks import build_scheduler
from src.utils.config import load_config
from src.utils.logger import TBLogger, get_logger
from src.utils.seed import set_seed


def build_optimizer(model, cfg):
    if hasattr(model, "encoder"):
        enc_params = list(model.encoder.parameters())
        dec_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
        groups = [
            {"params": [p for p in enc_params if p.requires_grad], "lr": cfg.train.encoder_lr},
            {"params": [p for p in dec_params if p.requires_grad], "lr": cfg.train.decoder_lr},
        ]
    else:
        # PlainUNet etc.: no encoder/decoder split, use single lr
        groups = [{"params": [p for p in model.parameters() if p.requires_grad],
                   "lr": getattr(cfg.train, "lr", 0.001)}]
    return torch.optim.AdamW(groups, weight_decay=cfg.train.weight_decay)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--base", default=None)
    ap.add_argument("--name", default=None,
                    help="experiment name -> outputs/runs/<name>/{checkpoints,logs}")
    ap.add_argument("--split-tag", default="",
                    help="e.g. 'fold0' -> use train_fold0.csv / val_fold0.csv")
    ap.add_argument("--seed", type=int, default=None,
                    help="覆盖 cfg.project.seed；用于同一配置跑多个种子，测单次训练的随机波动")
    args = ap.parse_args()

    cfg = load_config(args.config, args.base)
    if args.name:
        run_dir = os.path.join(cfg.outputs.root, "runs", args.name)
        cfg.outputs.checkpoint_dir = os.path.join(run_dir, "checkpoints")
        cfg.outputs.log_dir = os.path.join(run_dir, "logs")
    if args.seed is not None:
        cfg.project.seed = args.seed
    set_seed(cfg.project.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(cfg.outputs.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.outputs.log_dir, exist_ok=True)
    logger = get_logger(log_file=os.path.join(cfg.outputs.log_dir, "train.log"))
    tb = TBLogger(cfg.outputs.log_dir)

    train_csv = f"train_{args.split_tag}.csv" if args.split_tag else "train.csv"
    val_csv = f"val_{args.split_tag}.csv" if args.split_tag else "val.csv"
    train_ds = LungNoduleDataset(cfg.data.npz_dir, os.path.join(cfg.data.split_dir, train_csv),
                                 cfg, phase="train")
    val_ds = LungNoduleDataset(cfg.data.npz_dir, os.path.join(cfg.data.split_dir, val_csv),
                               cfg, phase="val")
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True,
                              num_workers=cfg.data.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False,
                            num_workers=cfg.data.num_workers)

    model_type = getattr(cfg.model, "type", "segmenter")
    if model_type == "plain_unet":
        model = PlainUNet(in_channels=cfg.model.in_channels,
                          num_classes=cfg.model.num_classes,
                          base_channels=getattr(cfg.model, "base_channels", 64))
    else:
        model = Segmenter(backbone=cfg.model.backbone, in_channels=cfg.model.in_channels,
                          pretrained=cfg.model.pretrained, num_classes=cfg.model.num_classes,
                          freeze=cfg.model.freeze_stages)
    model.to(device)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))
    criterion = CombinedLoss(cfg)

    # 注意：plain_unet 的 config 只设 model.type，model.backbone 仍从 default.yaml
    # 继承为 resnet34，直接打印 backbone 会误报架构。以实际建模用的 type 为准。
    arch = model_type if model_type != "segmenter" else getattr(cfg.model, "backbone", "?")
    logger.info(f"train={len(train_ds)} val={len(val_ds)} device={device} "
                f"arch={arch} target={getattr(cfg.loss, 'target', '?')} "
                f"seed={cfg.project.seed} split_tag={args.split_tag or 'none'}")
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, scheduler,
                      cfg, device, logger, tb)
    trainer.fit()
    tb.close()


if __name__ == "__main__":
    main()
