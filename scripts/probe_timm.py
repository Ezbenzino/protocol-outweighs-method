# -*- coding: utf-8 -*-
"""探测：哪些 Transformer 骨干能在 128x128 输入下以 features_only 模式工作。

不猜模型名，逐个真试。要求：
  1) 能以 features_only=True 建出来
  2) 128x128 输入能前向
  3) 给出各层通道数与下采样倍率（写编码器适配层要用）
"""
import ctypes, os, sys
for p in (os.path.join(sys.prefix, "Library", "bin", "libiomp5md.dll"),
          os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib", "libiomp5md.dll")):
    if os.path.exists(p):
        try: ctypes.WinDLL(p); break
        except OSError: pass
import torch
try:
    import timm
    print("timm", timm.__version__)
except ImportError:
    raise SystemExit("timm 未安装。先跑：pip install timm")

CANDS = ["pvt_v2_b1", "pvt_v2_b2", "mit_b1", "swin_tiny_patch4_window7_224",
         "swinv2_tiny_window8_256", "maxvit_tiny_tf_224", "convnext_tiny"]
x = torch.zeros(2, 3, 128, 128)
for name in CANDS:
    try:
        m = timm.create_model(name, features_only=True, pretrained=False)
        m.eval()
        with torch.no_grad():
            fs = m(x)
        shapes = [tuple(f.shape[1:]) for f in fs]
        print(f"  [OK]   {name:<32} 通道 {m.feature_info.channels()}  "
              f"倍率 {m.feature_info.reduction()}  实际 {shapes}")
    except Exception as e:
        print(f"  [失败] {name:<32} {type(e).__name__}: {str(e)[:110]}")
