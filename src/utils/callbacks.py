"""Early stopping and learning-rate scheduling."""
import math

import torch


class EarlyStopping:
    """Stop training when a monitored metric stops improving (higher is better)."""

    def __init__(self, patience=15, mode="max", min_delta=0.0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = None
        self.best_epoch = -1
        self.counter = 0

    def step(self, metric, epoch):
        """Return (should_stop, is_best)."""
        if self.best is None:
            self.best = metric
            self.best_epoch = epoch
            return False, True
        delta = metric - self.best if self.mode == "max" else self.best - metric
        if delta > self.min_delta:
            self.best = metric
            self.best_epoch = epoch
            self.counter = 0
            return False, True
        self.counter += 1
        return self.counter >= self.patience, False


def build_scheduler(optimizer, cfg, steps_per_epoch):
    """Warmup + cosine annealing scheduler driven by global optimizer steps."""
    epochs = cfg.train.epochs
    warmup_epochs = cfg.train.warmup_epochs
    min_lr = cfg.train.min_lr
    base_lr = cfg.train.lr
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr / base_lr, cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
