# -*- coding: utf-8 -*-
"""把 data/processed/npz 下的 npz 从"未压缩存储"改成"无损压缩"，原地替换。

【为什么能省这么多】
这些文件是 np.savez 写的，压缩方式是 stored（完全不压缩）。里面两个数组：
  slices     (N,512,512) uint8  CT 灰度
  consensus  (N,512,512) uint8  票数 0..4，绝大部分是 0
票数图几乎全零，deflate 后接近消失；CT 灰度也有可观冗余。实测平均 3.84 倍，
120 GB 可降到约 31 GB。

【为什么是无损的】
np.savez_compressed 用的是 zip deflate，与 np.savez 的差别只在压缩方式，
数组的 dtype、shape、数值逐字节不变。本脚本每写完一个文件都【读回来跟原数组
逐元素比对】，不通过就不替换。

【为什么不会影响已有代码】
读取端一律用 np.load，numpy 自动识别压缩方式，调用代码一个字都不用改。
唯一的代价是每次读多花一点解压时间（每例约零点几秒）。
注意：这些文件本来就不支持 mmap（.npz 是 zip 归档，mmap_mode 对它无效），
所以并没有失去任何原有能力。

【一个必须避开的坑】
np.savez_compressed 会给不以 .npz 结尾的路径【自动补上 .npz 后缀】。
若写成 np.savez_compressed(f + ".tmp", ...)，实际落盘的是 f + ".tmp.npz"，
随后 os.replace(f + ".tmp", f) 就会找不到文件——而那时原文件已经被读过、
临时文件已经写出，处于最容易搞混的状态。这里改为传【已打开的文件句柄】，
numpy 对句柄不做任何改名，路径完全由本脚本掌控。

【中断安全】
先写 <名字>.tmp，校验通过后才用 os.replace 原子替换。中途断电最多留下一个
.tmp 垃圾文件，原文件绝不会损坏。已经压过的文件会被跳过，可以反复运行。

用法（PowerShell，在仓库根目录下）：
    python scripts\\compress_npz.py --dry-run     # 先看能省多少，不动文件
    python scripts\\compress_npz.py               # 真的压
不占 GPU，可以和训练同时跑。
"""
import argparse
import glob
import os
import time
import zipfile

import numpy as np


def is_compressed(path):
    """已经压过的文件跳过。整个归档里只要还有 stored 的成员就算没压过。"""
    try:
        with zipfile.ZipFile(path) as z:
            infos = z.infolist()
            return bool(infos) and all(i.compress_type != zipfile.ZIP_STORED for i in infos)
    except Exception:                                        # noqa: BLE001
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/processed/npz")
    ap.add_argument("--dry-run", action="store_true", help="只统计，不写任何文件")
    ap.add_argument("--limit", type=int, default=0, help=">0 时只处理前 N 个，用于试水")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*.npz")))
    if args.limit:
        files = files[:args.limit]
    todo = [f for f in files if not is_compressed(f)]
    total = sum(os.path.getsize(f) for f in files)
    print(f"目录 {args.dir}")
    print(f"  文件 {len(files)} 个，合计 {total/1e9:.1f} GB")
    print(f"  已压缩 {len(files)-len(todo)} 个，待压缩 {len(todo)} 个")
    if not todo:
        print("  没有需要处理的文件。")
        return
    if args.dry_run:
        print("\n[试算] 取前 3 个实测压缩率后外推：")
        r_in = r_out = 0.0
        for f in todo[:3]:
            a = np.load(f)
            tmp = f + ".dryrun"
            with open(tmp, "wb") as fh:          # 传句柄，numpy 不会自动补 .npz
                np.savez_compressed(fh, **{k: a[k] for k in a.files})
            a.close()
            r_in += os.path.getsize(f)
            r_out += os.path.getsize(tmp)
            os.remove(tmp)
        ratio = r_in / max(r_out, 1)
        print(f"  实测压缩率 {ratio:.2f}x")
        print(f"  预计 {total/1e9:.1f} GB -> {total/1e9/ratio:.1f} GB，"
              f"可释放约 {total/1e9*(1-1/ratio):.0f} GB")
        print("  [试算完成，未修改任何文件]")
        return

    t0 = time.time()
    saved = 0
    for i, f in enumerate(todo, 1):
        before = os.path.getsize(f)
        tmp = f + ".tmp"
        try:
            a = np.load(f)
            arrays = {k: a[k] for k in a.files}
            with open(tmp, "wb") as fh:          # 传句柄，numpy 不会自动补 .npz
                np.savez_compressed(fh, **arrays)
            # 逐元素校验：不通过就绝不替换
            b = np.load(tmp)
            if set(b.files) != set(arrays) or not all(
                    np.array_equal(b[k], arrays[k]) for k in arrays):
                b.close()
                os.remove(tmp)
                print(f"  [跳过] {os.path.basename(f)} 校验失败，原文件保持不动")
                continue
            b.close()
            a.close()
            after = os.path.getsize(tmp)
            os.replace(tmp, f)                # 原子替换，同盘操作
            saved += before - after
        except Exception as e:                                # noqa: BLE001
            if os.path.exists(tmp):
                os.remove(tmp)
            print(f"  [跳过] {os.path.basename(f)}  {type(e).__name__}: {str(e)[:60]}")
            continue
        if i % 25 == 0 or i == len(todo):
            el = time.time() - t0
            eta = (len(todo) - i) * el / i / 60
            print(f"  {i}/{len(todo)}  已释放 {saved/1e9:.1f} GB  "
                  f"用时 {el/60:.1f} 分钟  剩余约 {eta:.1f} 分钟", flush=True)
    print(f"\n完成。释放 {saved/1e9:.1f} GB，用时 {(time.time()-t0)/60:.1f} 分钟。")


if __name__ == "__main__":
    main()
