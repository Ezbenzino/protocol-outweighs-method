# -*- coding: utf-8 -*-
"""建模型、数参数、试前向、量显存——在投入几小时训练【之前】跑。

论文的架构表需要参数量这一列：如果新架构参数量比 ResNet-34 大一倍，
那"架构效应"里就混进了"模型规模效应"，两者分不开，对照就不干净。
所以这个脚本同时充当"容量可比"的核对。

用法：
    python scripts\\check_arch.py
    python scripts\\check_arch.py --batch 32          # 试真实训练 batch 是否 OOM
"""
import ctypes, os, sys
for _p in (os.path.join(sys.prefix, "Library", "bin", "libiomp5md.dll"),
           os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib", "libiomp5md.dll")):
    if os.path.exists(_p):
        try:
            ctypes.WinDLL(_p); break
        except OSError:
            pass

import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.segmenter import Segmenter
from src.models.plain_unet import PlainUNet

BACKBONES = ["resnet18", "resnet34", "convnext_tiny", "pvt_v2_b1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--no-pretrained", action="store_true",
                    help="跳过权重下载，只验证结构能建能跑")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev}  batch={args.batch}  input={args.size}x{args.size}")
    if dev.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}  "
              f"显存 {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print("=" * 88)
    print(f"{'架构':<16}{'参数量':>12}{'编码器通道':>22}{'倍率':>7}{'前向':>8}{'训练峰值显存':>14}")
    print("-" * 88)

    rows = []
    for b in BACKBONES + ["plain_unet"]:
        try:
            if b == "plain_unet":
                m = PlainUNet(in_channels=3, num_classes=1, base_channels=64)
                chans, scale = "-", "-"
            else:
                m = Segmenter(backbone=b, in_channels=3, pretrained=not args.no_pretrained,
                              num_classes=1)
                chans = str(m.encoder.skip_channels)
                scale = m.encoder.final_scale
            n = sum(p.numel() for p in m.parameters())
            m.to(dev).train()
            if dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            x = torch.randn(args.batch, 3, args.size, args.size, device=dev)
            # 前向 + 反向：只测前向会低估训练显存，激活值才是大头
            out = m(x)
            reg = out[0] if isinstance(out, (tuple, list)) else out
            reg.float().pow(2).mean().backward()
            peak = torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else float("nan")
            print(f"{b:<16}{n/1e6:>10.2f}M{chans:>22}{str(scale):>7}"
                  f"{str(tuple(reg.shape[1:])):>8}{peak:>12.2f} GB")
            rows.append((b, n, peak))
            del m, x, out, reg
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:                                   # noqa: BLE001
            print(f"{b:<16}  失败: {type(e).__name__}: {str(e)[:60]}")
    print("-" * 88)
    if rows:
        ns = np.array([r[1] for r in rows], float)
        print(f"参数量跨度 {ns.min()/1e6:.1f}M ~ {ns.max()/1e6:.1f}M （最大/最小 = {ns.max()/ns.min():.1f}x）")
        print("[判读] 若某一臂的参数量远离其余，架构效应里就混进了模型规模效应，")
        print("       论文里必须把参数量这一列一起报出来，让读者自己判断。")
        ok = [r for r in rows if r[2] == r[2]]
        if ok:
            print(f"训练峰值显存最大 {max(r[2] for r in ok):.2f} GB —— "
                  f"超过可用显存就要用梯度累积保持等效 batch，不能直接改小 batch")


if __name__ == "__main__":
    main()
