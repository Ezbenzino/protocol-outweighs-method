"""训练前的自检：确认四个训练靶各自用对了监督信号，且 CSL 泄漏已修复。

**在启动 20 小时训练之前先跑这个**。它在几秒内就能发现配置或代码接错的问题，
避免跑完一整夜才发现某个臂的靶是错的。

三项检查：
  ① 四个训练靶在同一批真实数据上的损失值应当互不相同（说明靶确实切换了）
  ② union / majority 两个模式下，把 sample["consensus"] 换成随机值，
     损失必须【完全不变】—— 这是 CSL 泄漏是否修复的判据。
     旧版 combo.py 无条件读 consensus，此项必然失败。
  ③ consensus / soft 两个模式下，同样的扰动必须【改变】损失（它们本来就该用软概率）

用法：
    python scripts\\verify_targets.py
"""
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from src.data.dataset import LungNoduleDataset
from src.losses.combo import CombinedLoss, TARGET_SPECS
from src.utils.config import load_config

CFG = "configs/default.yaml"
TOL = 1e-9


def losses_for(cfg, target, batch, region, boundary, use_csl):
    c = copy.deepcopy(cfg)
    c.loss.target = target
    c.loss.use_dice = True
    c.loss.use_bce = True
    c.loss.use_sitl = False
    c.loss.use_csl = use_csl
    c.loss.use_brbc = False
    c.loss.use_lovasz = False
    total, parts = CombinedLoss(c)(region, boundary, batch)
    return float(total), {k: float(v) for k, v in parts.items()}


def main():
    cfg = load_config(CFG)
    ds = LungNoduleDataset(cfg.data.npz_dir,
                           os.path.join(cfg.data.split_dir, "val_fold0.csv"),
                           cfg, phase="val")
    batch = next(iter(DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)))
    torch.manual_seed(0)
    region = torch.randn_like(batch["union_mask"])      # 随机 logits 即可，只验证接线
    boundary = torch.randn_like(batch["union_mask"])

    print(f"数据：val_fold0 前 8 个样本  patch={tuple(batch['union_mask'].shape[-2:])}")
    print(f"GT 面积均值  V>=1(union)={batch['union_mask'].sum(dim=(1,2,3)).mean():.1f}  "
          f"V>=2(majority)={batch['majority_mask'].sum(dim=(1,2,3)).mean():.1f}  "
          f"p=V/4 概率和={batch['consensus'].sum(dim=(1,2,3)).mean():.1f}")

    # ---- 检查 ① 四个靶应当给出不同的损失 ----
    print("\n① 四个训练靶的损失值（应互不相同）")
    print(f"{'target':<12}{'total':>10}{'dice':>10}{'bce':>10}   spec")
    vals = {}
    for t in TARGET_SPECS:
        tot, parts = losses_for(cfg, t, batch, region, boundary, use_csl=False)
        vals[t] = tot
        s = TARGET_SPECS[t]
        print(f"{t:<12}{tot:>10.5f}{parts['dice']:>10.5f}{parts['bce']:>10.5f}   "
              f"dice<-{s['dice']}, bce<-{s['bce']}")
    if len(set(round(v, 6) for v in vals.values())) < len(vals):
        print("  [失败] 有两个训练靶给出了相同的损失，说明靶没有真正切换。")
    else:
        print("  [通过] 四个靶互不相同。")

    # ---- 检查 ②③ 扰动 consensus，看哪些靶会受影响 ----
    print("\n②③ 把 sample['consensus'] 换成随机值后，损失是否改变（含 CSL）")
    print(f"{'target':<12}{'原损失':>12}{'扰动后':>12}{'变化':>12}   判定")
    perturbed = {k: v for k, v in batch.items()}
    torch.manual_seed(1)
    perturbed["consensus"] = torch.rand_like(batch["consensus"])

    ok = True
    for t in TARGET_SPECS:
        a, _ = losses_for(cfg, t, batch, region, boundary, use_csl=True)
        b, _ = losses_for(cfg, t, perturbed, region, boundary, use_csl=True)
        changed = abs(a - b) > TOL
        should_change = TARGET_SPECS[t]["bce"] == "consensus"
        good = (changed == should_change)
        ok &= good
        verdict = ("[通过] " if good else "[失败] ") + \
                  ("应改变且已改变" if should_change and changed else
                   "应改变但没变" if should_change else
                   "不应改变但变了 <- CSL 泄漏未修复" if changed else "不应改变且未变")
        print(f"{t:<12}{a:>12.5f}{b:>12.5f}{b - a:>+12.5f}   {verdict}")

    print()
    if ok:
        print("全部通过。CSL 泄漏已修复，四个训练靶接线正确，可以开始训练。")
    else:
        print("存在失败项，请勿开始训练——先检查 src/losses/combo.py 是否已替换为新版。")
        sys.exit(1)


if __name__ == "__main__":
    main()
