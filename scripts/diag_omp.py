# -*- coding: utf-8 -*-
"""定位 OMP Error #15：进程里到底加载了几份 libiomp5md.dll，分别来自哪个目录。

我先后给出过两个猜测（scipy/skimage、导入顺序），都被实测推翻了。不再猜，直接量。

做法：调用 Windows 的 EnumProcessModules，把当前进程已加载的 DLL 全路径列出来，
在每一步 import 前后各查一次，看是哪一次 import 把第二份 OpenMP 运行时拉进来的。

本脚本【故意】设置 KMP_DUPLICATE_LIB_OK=TRUE。这一点必须说清楚：
该变量会让"两份 OpenMP"从致命错误降级为继续执行，Intel 明确警告它可能静默算错。
但本脚本【不产出任何研究数字】，只列举 DLL 路径，算错也无从算起；
设它只是为了让进程别在第一次冲突时就死掉，好把后面的 import 全部走完。
真正的实验脚本一律不设这个变量。

用法（PowerShell，在仓库根目录下，几秒钟）：
    python scripts\\diag_omp.py
"""
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # 仅为让枚举跑完，见上方说明

import ctypes
import sys
from ctypes import wintypes

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_psapi = ctypes.WinDLL("psapi", use_last_error=True)
_k32 = ctypes.WinDLL("kernel32", use_last_error=True)


def loaded_dlls():
    """列出当前进程已加载的全部 DLL 全路径。"""
    h = _k32.GetCurrentProcess()
    arr = (ctypes.c_void_p * 8192)()
    need = wintypes.DWORD()
    if not _psapi.EnumProcessModules(h, ctypes.byref(arr), ctypes.sizeof(arr),
                                     ctypes.byref(need)):
        return []
    n = min(need.value // ctypes.sizeof(ctypes.c_void_p), 8192)
    buf = ctypes.create_unicode_buffer(2048)
    out = []
    for i in range(n):
        if _psapi.GetModuleFileNameExW(h, arr[i], buf, 2048):
            out.append(buf.value)
    return out


KEY = ("iomp", "omp140", "vcomp", "gomp", "mkl_core", "mkl_rt", "openblas")


def snap():
    return {p for p in loaded_dlls() if any(k in os.path.basename(p).lower() for k in KEY)}


def step(tag, fn):
    before = snap()
    try:
        fn()
        err = None
    except Exception as e:                                  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    after = snap()
    new = sorted(after - before)
    mark = "  <== 新增" if new else ""
    print(f"\n[{tag}]{'  失败: ' + err if err else ''}{mark}")
    for p in new:
        print(f"    + {p}")
    if not new and not err:
        print("    (无新增 OpenMP / BLAS 运行时)")
    sys.stdout.flush()


print("=" * 96)
print("OMP Error #15 定位：逐步 import，观察 OpenMP / BLAS 运行时的加载来源")
print("=" * 96)
print(f"python : {sys.executable}")
print(f"prefix : {sys.prefix}")
step("起始", lambda: None)


def _numpy():
    global np
    import numpy as np


def _torch():
    global torch
    import torch


def _scipy():
    import scipy.ndimage  # noqa: F401


def _skimage():
    import skimage.measure  # noqa: F401


def _tv():
    import torchvision  # noqa: F401


def _timm():
    import timm  # noqa: F401


def _models():
    from src.models.segmenter import Segmenter  # noqa: F401
    from src.models.plain_unet import PlainUNet  # noqa: F401


def _cuda():
    torch.zeros(8, device="cuda" if torch.cuda.is_available() else "cpu").sum()


for tag, fn in [("import numpy", _numpy), ("import torch", _torch),
                ("import torchvision", _tv), ("import timm", _timm),
                ("import scipy.ndimage", _scipy), ("import skimage.measure", _skimage),
                ("import src.models", _models), ("torch 首次运算", _cuda)]:
    step(tag, fn)

print("\n" + "=" * 96)
print("最终：进程内所有 OpenMP / BLAS 运行时")
print("=" * 96)
final = sorted(snap())
for p in final:
    print(f"  {p}")
base = {}
for p in final:
    base.setdefault(os.path.basename(p).lower(), []).append(p)
dup = {b: v for b, v in base.items() if len(v) > 1}
print("\n同名多份的：", "无" if not dup else "")
for b, v in dup.items():
    print(f"  {b}  x{len(v)}")
    for p in v:
        print(f"      {p}")

print("\n" + "-" * 96)
print("磁盘上 libiomp5md.dll 的所有副本（决定怎么修的关键）：")
import glob
roots = [sys.prefix, os.path.join(sys.prefix, "Library", "bin"),
         os.path.join(sys.prefix, "Lib", "site-packages")]
seen = set()
for r in roots:
    for p in glob.glob(os.path.join(r, "**", "libiomp5md.dll"), recursive=True):
        if p.lower() in seen:
            continue
        seen.add(p.lower())
        print(f"  {os.path.getsize(p)/1e6:7.2f} MB   {p}")
if not seen:
    print("  (在 env 目录下没找到，可能来自系统 PATH 上的其他软件)")
