"""Training loop with AMP, gradient accumulation, scheduler and early stopping."""
import gc
import os
import time

import torch
from torch.cuda.amp import GradScaler, autocast

from src.engine.evaluator import evaluate
from src.utils.callbacks import EarlyStopping


def _to_device(sample, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in sample.items()}


class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer,
                 scheduler, cfg, device, logger, tb):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg
        self.device = device
        self.logger = logger
        self.tb = tb

        self.epochs = cfg.train.epochs
        self.accum = cfg.train.grad_accum_steps
        self.grad_clip = float(getattr(cfg.train, "grad_clip_norm", 0.0))
        self.use_amp = cfg.train.use_amp
        self.amp_dtype = torch.bfloat16 if cfg.train.amp_dtype == "bf16" else torch.float16
        self.scaler = GradScaler(enabled=(self.use_amp and cfg.train.amp_dtype != "bf16"))
        self.es = EarlyStopping(cfg.train.early_stop_patience, mode="max")
        self.ckpt_dir = cfg.outputs.checkpoint_dir
        self.log_every = cfg.train.log_every
        self.val_every = cfg.train.val_every
        self._step = 0

    def _run_epoch(self):
        self.model.train()
        total_loss = 0.0
        n = 0
        self.optimizer.zero_grad()
        n_batches = len(self.train_loader)

        for step, sample in enumerate(self.train_loader):
            sample = _to_device(sample, self.device)
            image = sample["image"]
            with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                region, boundary = self.model(image)
                loss, _ = self.criterion(region, boundary, sample)

            loss = loss / self.accum
            if torch.isnan(loss) or torch.isinf(loss):
                self.logger.warning(f"  NaN/Inf loss at step {step+1}, skipping backward")
                self.optimizer.zero_grad()
                continue
            self.scaler.scale(loss).backward()

            if (step + 1) % self.accum == 0 or (step + 1) == n_batches:
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                if self.scheduler is not None:
                    self.scheduler.step()

            total_loss += loss.item() * self.accum
            n += 1
            self._step += 1
            if (step + 1) % self.log_every == 0:
                self.logger.info(f"  step {step + 1}/{n_batches} loss={total_loss / n:.4f}")
                self.tb.log_scalar("train/loss_step", total_loss / n, self._step)
        return total_loss / max(1, n)

    def fit(self):
        monitor_name = self.cfg.train.early_stop_metric
        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_loss = self._run_epoch()
            self.tb.log_scalar("train/loss", train_loss, epoch)
            self.tb.log_scalar("train/lr", self.optimizer.param_groups[0]["lr"], epoch)
            self.logger.info(
                f"Epoch {epoch}/{self.epochs} | train_loss={train_loss:.4f} | {time.time() - t0:.1f}s")

            if epoch % self.val_every == 0:
                metrics = evaluate(self.model, self.val_loader, self.device,
                                   self.cfg.eval.thresholds,
                                   activation=getattr(self.cfg.model,
                                                      "final_activation", "sigmoid"))
                for k, v in metrics.items():
                    if isinstance(v, float):
                        self.tb.log_scalar(f"val/{k}", v, epoch)
                self.logger.info(
                    "  val | " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()
                                           if isinstance(v, float)))
                monitor = metrics.get(monitor_name, metrics.get("dice", 0.0))
                stop, is_best = self.es.step(monitor, epoch)
                if is_best:
                    self._save("best.pth")
                    self.logger.info(f"  saved best (epoch {epoch}, {monitor_name}={monitor:.4f})")
                if stop:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
            self._save("last.pth")
            # Memory cleanup to prevent leaks across epochs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _save(self, name):
        path = os.path.join(self.ckpt_dir, name)
        # meta 里记下输出活化方式。评测端据此还原概率，不再假设一定是 sigmoid。
        # 老 checkpoint 没有这个字段，读取端一律按 sigmoid 处理，行为不变。
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "meta": {"final_activation": getattr(self.cfg.model, "final_activation",
                                                 "sigmoid")},
        }, path)
